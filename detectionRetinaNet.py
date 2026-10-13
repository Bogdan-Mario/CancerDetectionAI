"""
RetinaNet + EfficientNet-B5 (Mammo-CLIP weights) for microcalcification DETECTION.
mask -> cluster boxes -> patch training -> per-fold (folds.csv) -> FROC + mAP@50 + mAP@90.
"""
import matplotlib; matplotlib.use("Agg")
import os, glob, json, random, functools
from collections import OrderedDict
import numpy as np, pandas as pd, cv2, torch
import torch.nn as nn
from scipy import ndimage
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import RetinaNet
from torchvision.ops import FeaturePyramidNetwork, box_iou, nms
from torchvision.ops.feature_pyramid_network import LastLevelP6P7
from efficientnet_pytorch import EfficientNet
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm

print = functools.partial(print, flush=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# config------------------------
SOURCE_IMG_DIR  = "/CompletePipeline/Normal"
SOURCE_MASK_DIR = "/Masks/Calcification_Only"
FOLDS_CSV   = "folds.csv"
FOLDS       = [1, 2, 3, 4, 5]
ENCODER_INIT = "mammoclip"            # "mammoclip" | "imagenet"
MAMMO_CKPT  = "/checkpoints/Pre-trained-checkpoints/b5-model-best-epoch-7.tar"
BACKBONE    = "efficientnet-b5"
OUTPUT_BASE = f"Experiments_RetinaNet_{ENCODER_INIT}_b5"

PATCH_SIZE   = 256
PATCH_STRIDE = 256
DILATE       = 15          # px dilation to merge nearby calcs into a cluster box
MIN_BOX      = 3           # ignore boxes smaller than this
NEG_PER_POS  = 1.0         # negative patches per positive in TRAIN
FROC_IOU     = 0.2         # a GT cluster counts as detected at IoU >= this
BATCH        = 4
EPOCHS       = 50
PATIENCE     = 10
LR           = 1e-4
VAL_FRACTION = 0.1
NUM_WORKERS  = 2
SEED         = 42
NORM_MEAN, NORM_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


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
    _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    return m

def pad(a, P):
    h, w = a.shape
    if h < P or w < P:
        a = np.pad(a, ((0, max(0, P - h)), (0, max(0, P - w))), constant_values=0)
    return a

@functools.lru_cache(maxsize=8)
def image_boxes(iid):
    """Full-image cluster boxes [x1,y1,x2,y2] from the calc mask (dilate -> CC)."""
    m = _mask(iid)
    if m is None: return np.zeros((0, 4), np.float32)
    md = cv2.dilate(m, np.ones((DILATE, DILATE), np.uint8)) if DILATE > 0 else m
    lab, n = ndimage.label(md > 0)
    out = []
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        if x2 - x1 >= MIN_BOX and y2 - y1 >= MIN_BOX:
            out.append([x1, y1, x2, y2])
    return np.array(out, np.float32).reshape(-1, 4)


def build_train_index(ids):
    pos, neg = [], []
    for iid in ids:
        img = _img(iid); H, W = img.shape
        boxes = image_boxes(iid)
        for py in range(0, H, PATCH_STRIDE):
            for px in range(0, W, PATCH_STRIDE):
                if np.count_nonzero(img[py:py+PATCH_SIZE, px:px+PATCH_SIZE]) < 1:
                    continue
                local = []
                for (x1, y1, x2, y2) in boxes:
                    ix1, iy1 = max(x1, px), max(y1, py)
                    ix2, iy2 = min(x2, px+PATCH_SIZE), min(y2, py+PATCH_SIZE)
                    if ix2 - ix1 > MIN_BOX and iy2 - iy1 > MIN_BOX:
                        local.append([ix1-px, iy1-py, ix2-px, iy2-py])
                (pos if local else neg).append((iid, px, py, local))
    random.Random(SEED).shuffle(neg)
    return pos + neg[:int(len(pos) * NEG_PER_POS)]


def aug_det(img, boxes):
    H, W = img.shape
    if random.random() < 0.5:
        img = np.fliplr(img).copy()
        if len(boxes): boxes[:, [0, 2]] = W - boxes[:, [2, 0]]
    if random.random() < 0.5:
        img = np.flipud(img).copy()
        if len(boxes): boxes[:, [1, 3]] = H - boxes[:, [3, 1]]
    return img, boxes

class DetDS(Dataset):
    def __init__(self, index, train=False): self.index = index; self.train = train
    def __len__(self): return len(self.index)
    def __getitem__(self, i):
        iid, px, py, local = self.index[i]
        patch = pad(_img(iid)[py:py+PATCH_SIZE, px:px+PATCH_SIZE], PATCH_SIZE)
        boxes = np.array(local, np.float32).reshape(-1, 4)
        if self.train:
            patch, boxes = aug_det(patch, boxes)
        x = torch.from_numpy(np.stack([patch]*3, 0).astype(np.float32) / 255.0)   # RetinaNet transform normalizes
        target = {"boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                  "labels": torch.ones((len(boxes),), dtype=torch.int64)}
        return x, target

def collate(b): return tuple(zip(*b))


# model
class EffFPN(nn.Module):
    def __init__(self, body, out_channels=256):
        super().__init__()
        self.body = body
        self.keys = ["reduction_3", "reduction_4", "reduction_5"]   # strides 8,16,32
        with torch.no_grad():
            eps = body.extract_endpoints(torch.zeros(1, 3, PATCH_SIZE, PATCH_SIZE))
        in_ch = [eps[k].shape[1] for k in self.keys]
        self.fpn = FeaturePyramidNetwork(in_ch, out_channels,
                                         extra_blocks=LastLevelP6P7(in_ch[-1], out_channels))
        self.out_channels = out_channels
    def forward(self, x):
        eps = self.body.extract_endpoints(x)
        return self.fpn(OrderedDict((str(i), eps[k]) for i, k in enumerate(self.keys)))

def _load_subset(effnet, src):
    tgt = effnet.state_dict()
    ok = {k: v for k, v in src.items() if k in tgt and tgt[k].shape == v.shape}
    effnet.load_state_dict(ok, strict=False)
    return len(ok), len(tgt)

def load_mammo_into_effnet(effnet, ckpt):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck.get("state_dict", ck))
    src = {}
    for k, v in sd.items():
        if "image_encoder." in k:
            src[k.split("image_encoder.", 1)[1]] = v
    n, tot = _load_subset(effnet, src)
    print(f"[Mammo-CLIP] matched {n}/{tot} backbone tensors")
    if n == 0: raise RuntimeError("no Mammo-CLIP weights matched")

def build_model():
    body = EfficientNet.from_name(BACKBONE, in_channels=3, image_size=None)
    model = RetinaNet(EffFPN(body), num_classes=2, min_size=PATCH_SIZE, max_size=PATCH_SIZE,
                      image_mean=list(NORM_MEAN), image_std=list(NORM_STD),
                      score_thresh=0.05, detections_per_img=300)
    if ENCODER_INIT == "mammoclip":
        load_mammo_into_effnet(model.backbone.body, MAMMO_CKPT)
    elif ENCODER_INIT == "imagenet":
        n, tot = _load_subset(model.backbone.body, EfficientNet.from_pretrained(BACKBONE, in_channels=3).state_dict())
        print(f"[ImageNet] matched {n}/{tot} backbone tensors")
    return model.to(DEVICE)


# evaluation
@torch.no_grad()
def eval_patch_map50(model, loader):
    model.eval()
    metric = MeanAveragePrecision(iou_thresholds=[0.5])
    for images, targets in loader:
        preds = model([im.to(DEVICE) for im in images])
        metric.update([{k: v.cpu() for k, v in p.items()} for p in preds], list(targets))
    return float(metric.compute()["map"])

@torch.no_grad()
def predict_image(model, iid):
    model.eval()
    img = _img(iid); H, W = img.shape
    coords = [(px, py) for py in range(0, H, PATCH_STRIDE) for px in range(0, W, PATCH_STRIDE)]
    bxs, scs = [], []
    for i in range(0, len(coords), BATCH):
        chunk = coords[i:i+BATCH]
        imgs = [torch.from_numpy(np.stack([pad(img[py:py+PATCH_SIZE, px:px+PATCH_SIZE], PATCH_SIZE)]*3, 0)
                                 .astype(np.float32)/255.0).to(DEVICE) for (px, py) in chunk]
        for (px, py), p in zip(chunk, model(imgs)):
            b, s = p["boxes"].cpu(), p["scores"].cpu()
            if len(b):
                b[:, [0, 2]] += px; b[:, [1, 3]] += py
                bxs.append(b); scs.append(s)
    if bxs:
        boxes, scores = torch.cat(bxs), torch.cat(scs)
        keep = nms(boxes, scores, 0.3)
        return boxes[keep], scores[keep]
    return torch.zeros((0, 4)), torch.zeros((0,))

def froc_boxes(per_image, thresholds=np.linspace(0.05, 0.95, 19),
               report_fp=(0.25, 0.5, 1.0, 2.0, 4.0)):
    total = sum(len(gt) for gt, _, _ in per_image); n_img = len(per_image)
    curve = []
    for t in thresholds:
        tp = fp = 0
        for gt, boxes, scores in per_image:
            pb = boxes[scores >= t]
            if len(gt) == 0:
                fp += len(pb); continue
            if len(pb) == 0: continue
            iou = box_iou(pb, gt)                        # (Np, Ng)
            hit = set()
            for j in range(len(pb)):
                m = (iou[j] >= FROC_IOU).nonzero().flatten().tolist()
                if m: hit.update(m)
                else: fp += 1
            tp += len(hit)
        curve.append((float(t), tp/total if total else 0.0, fp/n_img))
    arr = np.array([(s, f) for _, s, f in curve])
    sens_at = {}
    for target in report_fp:
        ok = arr[arr[:, 1] <= target]
        sens_at[f"sens@{target}fp"] = float(ok[:, 0].max()) if len(ok) else 0.0
    return curve, sens_at

@torch.no_grad()
def eval_full(model, ids):
    m50 = MeanAveragePrecision(iou_thresholds=[0.5]); m90 = MeanAveragePrecision(iou_thresholds=[0.9])
    per_image = []
    for iid in tqdm(ids, desc="eval"):
        boxes, scores = predict_image(model, iid)
        gt = torch.as_tensor(image_boxes(iid), dtype=torch.float32).reshape(-1, 4)
        preds = [{"boxes": boxes, "scores": scores, "labels": torch.ones(len(boxes), dtype=torch.int64)}]
        tgts  = [{"boxes": gt, "labels": torch.ones(len(gt), dtype=torch.int64)}]
        m50.update(preds, tgts); m90.update(preds, tgts)
        per_image.append((gt, boxes, scores))
    curve, sens_at = froc_boxes(per_image)
    return {"map50": float(m50.compute()["map"]), "map90": float(m90.compute()["map"]),
            **sens_at, "n_images": len(ids)}, curve


def load_fold(fold, seed=42):
    df = pd.read_csv(FOLDS_CSV); col = f"fold_{fold}"
    names = df["image_name"].astype(str)
    tr = names[df[col] == "train"].tolist(); va = names[df[col] == "val"].tolist(); te = names[df[col] == "test"].tolist()
    if not va:
        rng = random.Random(seed + fold); k = max(1, int(len(tr) * VAL_FRACTION))
        va = rng.sample(tr, k); vs = set(va); tr = [i for i in tr if i not in vs]
    return tr, va, te


def run_fold(fold, tr, va, te):
    run_dir = os.path.join(OUTPUT_BASE, f"fold{fold}"); os.makedirs(run_dir, exist_ok=True)
    print(f"\n===== fold {fold} | train {len(tr)} / val {len(va)} / test {len(te)} images =====")
    tr_idx, va_idx = build_train_index(tr), build_train_index(va)
    print(f"train patches {len(tr_idx)} | val patches {len(va_idx)}")
    tl = DataLoader(DetDS(tr_idx, True), batch_size=BATCH, shuffle=True,
                    num_workers=NUM_WORKERS, collate_fn=collate, drop_last=True)
    vl = DataLoader(DetDS(va_idx, False), batch_size=BATCH, shuffle=False,
                    num_workers=NUM_WORKERS, collate_fn=collate)

    model = build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", patience=4, factor=0.5)

    best, patience = -1.0, 0
    for epoch in range(EPOCHS):
        model.train(); running = 0.0
        for images, targets in tqdm(tl, desc=f"f{fold} Ep{epoch}", leave=False):
            images = [im.to(DEVICE) for im in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            loss = sum(model(images, targets).values())
            opt.zero_grad(); loss.backward(); opt.step()
            running += float(loss)
        vmap = eval_patch_map50(model, vl)
        sched.step(vmap)
        print(f"f{fold} Ep{epoch}: train_loss {running/len(tl):.4f} | val mAP@50 {vmap:.4f}")
        if vmap > best + 1e-4:
            best, patience = vmap, 0
            torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pth"))
        else:
            patience += 1
            if patience >= PATIENCE:
                print("early stop"); break

    model.load_state_dict(torch.load(os.path.join(run_dir, "best_model.pth"), map_location=DEVICE))
    metrics, curve = eval_full(model, te)
    metrics["fold"] = fold
    pd.DataFrame(curve, columns=["threshold", "sensitivity", "fp_per_image"]).to_csv(
        os.path.join(run_dir, "froc_curve.csv"), index=False)
    with open(os.path.join(run_dir, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"f{fold} TEST:", metrics)
    return metrics


def main():
    seed_all(SEED); os.makedirs(OUTPUT_BASE, exist_ok=True)
    results = [run_fold(fold, *load_fold(fold)) for fold in FOLDS]
    res = pd.DataFrame(results); res.to_csv(os.path.join(OUTPUT_BASE, "cv_results.csv"), index=False)
    print("\n===== CV summary =====")
    print(res.to_string(index=False))
    for k in ["map50", "map90", "sens@1.0fp"]:
        if k in res: print(f"{k}: {res[k].mean():.4f} +/- {res[k].std():.4f}")


if __name__ == "__main__":
    main()