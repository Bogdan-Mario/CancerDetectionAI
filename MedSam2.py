"""
Fine-tune MedSAM (SAM ViT-B) for box-prompted microcalcification segmentation.
Training prompt = GT cluster box (+jitter); freeze prompt encoder, train mask decoder
"""
import matplotlib; matplotlib.use("Agg")
import os, glob, json, random, functools
import numpy as np, pandas as pd, cv2, torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from segment_anything import sam_model_registry

print = functools.partial(print, flush=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- CONFIG ----------------
SOURCE_IMG_DIR  = "/Normal"
SOURCE_MASK_DIR = "/Masks/Calcification_Only"
MEDSAM_CKPT = "/checkpoints/sam_vit_b_01ec64.pth"           # download from bowang-lab/MedSAM
FOLDS_CSV   = "folds.csv"; FOLDS = [1, 2, 3, 4, 5]; VAL_FRACTION = 0.1
OUTPUT_BASE = "Experiments_MedSAM_boxprompt"
PATCH_SIZE  = 256
SAM_SIZE    = 1024
MASK_SIZE   = 256
BBOX_SHIFT  = 15            # px jitter on the GT box during training
FREEZE_IMAGE_ENCODER = True # True = train mask decoder only (light, low-mem); False = also tune ViT
BATCH = 2; EPOCHS = 50; PATIENCE = 8; LR = 1e-4; NUM_WORKERS = 2; SEED = 42; THRESHOLD = 0.5


def seed_all(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

@functools.lru_cache(maxsize=8)
def _img(iid):
    a = cv2.imread(os.path.join(SOURCE_IMG_DIR, iid + ".png"), cv2.IMREAD_GRAYSCALE)
    if a is None: raise FileNotFoundError(iid + ".png")
    return a
@functools.lru_cache(maxsize=8)
def _mask(iid):
    m = cv2.imread(os.path.join(SOURCE_MASK_DIR, iid + "_calc.png"), cv2.IMREAD_GRAYSCALE)
    if m is None: return None
    _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY); return m
def pad(a, P):
    h, w = a.shape
    if h < P or w < P: a = np.pad(a, ((0, max(0, P-h)), (0, max(0, P-w))), constant_values=0)
    return a

def load_fold(fold):
    df = pd.read_csv(FOLDS_CSV); col = f"fold_{fold}"; names = df["image_name"].astype(str)
    tr = names[df[col]=="train"].tolist(); va = names[df[col]=="val"].tolist(); te = names[df[col]=="test"].tolist()
    if not va:
        rng = random.Random(SEED+fold); k = max(1, int(len(tr)*VAL_FRACTION))
        va = rng.sample(tr, k); vs = set(va); tr = [i for i in tr if i not in vs]
    return tr, va, te

def build_samples(ids):
    S = []
    for iid in ids:
        img = _img(iid); m = _mask(iid)
        if m is None: continue
        H, W = img.shape
        for y in range(0, H, PATCH_SIZE):
            for x in range(0, W, PATCH_SIZE):
                mp = pad(m[y:y+PATCH_SIZE, x:x+PATCH_SIZE], PATCH_SIZE)
                ys, xs = np.where(mp > 127)
                if len(xs):
                    S.append((iid, x, y, [xs.min(), ys.min(), xs.max()+1, ys.max()+1]))
    return S

class MedSAMDS(Dataset):
    def __init__(self, samples, train=False): self.s = samples; self.train = train
    def __len__(self): return len(self.s)
    def __getitem__(self, i):
        iid, px, py, box = self.s[i]
        patch = pad(_img(iid)[py:py+PATCH_SIZE, px:px+PATCH_SIZE], PATCH_SIZE)
        mp = pad(_mask(iid)[py:py+PATCH_SIZE, px:px+PATCH_SIZE], PATCH_SIZE)
        box = np.array(box, np.float32)
        if self.train:
            box = box + np.random.uniform(-BBOX_SHIFT, BBOX_SHIFT, 4)
            box[:2] = np.clip(box[:2], 0, PATCH_SIZE-1); box[2:] = np.clip(box[2:], 1, PATCH_SIZE)
        img1024 = cv2.resize(patch, (SAM_SIZE, SAM_SIZE))
        x = np.stack([img1024]*3, 0).astype(np.float32)/255.0                   # [0,1], 3ch
        box1024 = box * (SAM_SIZE/PATCH_SIZE)
        mask256 = cv2.resize((mp > 127).astype(np.uint8), (MASK_SIZE, MASK_SIZE), interpolation=cv2.INTER_NEAREST)
        return (torch.from_numpy(x), torch.from_numpy(box1024).float(),
                torch.from_numpy(mask256[None].astype(np.float32)))

class MedSAM(torch.nn.Module):
    def __init__(self, sam):
        super().__init__()
        self.image_encoder = sam.image_encoder
        self.prompt_encoder = sam.prompt_encoder
        self.mask_decoder = sam.mask_decoder
        for p in self.prompt_encoder.parameters():
            p.requires_grad = False

    def forward(self, image, box):                       # image (B,3,1024,1024), box (B,4)
        ie = self.image_encoder(image)                   # (B,256,64,64) — batched, fine
        image_pe = self.prompt_encoder.get_dense_pe()
        outs = []
        for i in range(ie.shape[0]):                     
            with torch.no_grad():
                sparse, dense = self.prompt_encoder(points=None, boxes=box[i:i+1], masks=None)
            low, _ = self.mask_decoder(
                image_embeddings=ie[i:i+1], image_pe=image_pe,
                sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
                multimask_output=False)
            outs.append(low)
        return torch.cat(outs, 0)                         # (B,1,256,256)                                    # (B,1,256,256) logits

def dice_bce(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    num = 2*(p*target).sum((1,2,3)) + 1; den = p.sum((1,2,3)) + target.sum((1,2,3)) + 1
    return bce + (1 - (num/den).mean())

@torch.no_grad()
def eval_dice(model, loader):
    model.eval(); ds = []
    for x, box, y in loader:
        logits = model(x.to(DEVICE), box.to(DEVICE))
        p = (torch.sigmoid(logits) > THRESHOLD).float().cpu()
        for j in range(len(p)):
            inter = (p[j]*y[j]).sum(); s = p[j].sum()+y[j].sum()
            ds.append((2*inter/(s+1e-6)).item() if s > 0 else 1.0)
    return float(np.mean(ds)) if ds else 0.0

def run_fold(fold, tr, va, te):
    run_dir = os.path.join(OUTPUT_BASE, f"fold{fold}"); os.makedirs(run_dir, exist_ok=True)
    tr_s, va_s, te_s = build_samples(tr), build_samples(va), build_samples(te)
    print(f"\nfold {fold}: train {len(tr_s)}, val {len(va_s)}, test {len(te_s)} box samples")
    tl = DataLoader(MedSAMDS(tr_s, True), batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
    vl = DataLoader(MedSAMDS(va_s, False), batch_size=BATCH, num_workers=NUM_WORKERS)

    sam = sam_model_registry["vit_b"](checkpoint=MEDSAM_CKPT)
    model = MedSAM(sam).to(DEVICE)
    if FREEZE_IMAGE_ENCODER:
        for p in model.image_encoder.parameters(): p.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad]
    params_m = sum(p.numel() for p in params) / 1e6
    print(f"  params: {params_m:.1f}M trainable")
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=1e-4)

    best, patience = -1.0, 0
    for epoch in range(EPOCHS):
        model.train(); tot = 0.0
        for x, box, y in tqdm(tl, desc=f"f{fold} Ep{epoch}", leave=False):
            loss = dice_bce(model(x.to(DEVICE), box.to(DEVICE)), y.to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss)
        vd = eval_dice(model, vl)
        print(f"  fold {fold} ep {epoch}: loss {tot/len(tl):.4f}, dice {vd:.4f}")
        if vd > best + 1e-4:
            best, patience = vd, 0
            torch.save(model.state_dict(), os.path.join(run_dir, "medsam_best.pth"))
        else:
            patience += 1
            if patience >= PATIENCE: print("  stopping early"); break
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump({"fold": fold, "best_val_box_dice": best}, f, indent=2)
    return best

def main():
    seed_all(SEED); os.makedirs(OUTPUT_BASE, exist_ok=True)
    for fold in FOLDS:
        run_fold(fold, *load_fold(fold))

if __name__ == "__main__":
    main()