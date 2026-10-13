"""
Mammo-CLIP U-Net for microcalcification segmentation.
      MODE = "patch"    -> train + eval on 256 patches 
    MODE = "wholeimage" -> train + eval on downsized whole mammograms (baseline)
"""

import matplotlib
matplotlib.use("Agg")

import os
import glob
import json
import random
import functools
import numpy as np
import pandas as pd
import cv2
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from mammo_clip_unet import (MammoClipUNet, load_mammo_clip_encoder, DiceBCELoss,
                             dice_iou, BACKBONES, NORM_MEAN, NORM_STD)

print = functools.partial(print, flush=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# config
MODE       = "patch"                          # "patch" or "wholeimage"
BACKBONE   = "b5"                             # "b5" or "b2"
MAMMO_CKPT = "/checkpoints/Pre-trained-checkpoints/b5-model-best-epoch-7.tar"   # pre-trained checkpoint

SOURCE_IMG_DIR  = "/Normal"
SOURCE_MASK_DIR = "/Masks/Calcification_Only"
OUTPUT_BASE  = f"Experiments_MammoCLIP_{BACKBONE}_{MODE}"   
FOLDS_CSV    = "folds.csv"
FOLDS        = [1, 2]        # first two folds
VAL_FRACTION = 0.1          # carve val from each fold's train (folds.csv has only train/test)


PATCH_SIZE    = 256
PATCH_STRIDE  = 256           # 128 for 50% overlap
NEG_POS_RATIO = 2.0           # patch-mode train balancing
WHOLE_SIZE    = 1024          

BATCH_SIZE = 8 if MODE == "patch" else 2
EPOCHS     = 100
PATIENCE   = 15
LR         = 1e-4
FREEZE_ENCODER_WARMUP = 3     # epochs with encoder frozen 
NUM_WORKERS = 4
SEED = 42
THRESHOLD = 0.5


def seed_all(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@functools.lru_cache(maxsize=16)
def _img(iid):
    a = cv2.imread(os.path.join(SOURCE_IMG_DIR, iid + ".png"), cv2.IMREAD_GRAYSCALE)
    if a is None:
        raise FileNotFoundError(iid + ".png")
    return a

@functools.lru_cache(maxsize=16)
def _mask(iid):
    m = cv2.imread(os.path.join(SOURCE_MASK_DIR, iid + "_calc.png"), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    return m


def to_tensor(gray):
    x = np.stack([gray] * 3, 0).astype(np.float32) / 255.0
    for c in range(3):
        x[c] = (x[c] - NORM_MEAN[c]) / NORM_STD[c]
    return torch.from_numpy(x)

def pad(a, P):
    h, w = a.shape
    if h < P or w < P:
        a = np.pad(a, ((0, max(0, P - h)), (0, max(0, P - w))), constant_values=0)
    return a

def aug(img, m):
    if random.random() < 0.5: img, m = np.fliplr(img).copy(), np.fliplr(m).copy()
    if random.random() < 0.5: img, m = np.flipud(img).copy(), np.flipud(m).copy()
    k = random.randint(0, 3)
    if k: img, m = np.rot90(img, k).copy(), np.rot90(m, k).copy()
    return img, m


# datasets
def build_patch_index(image_ids, balance=False):
    recs = []
    for iid in image_ids:
        img, m = _img(iid), _mask(iid)
        H, W = img.shape
        for y in range(0, H, PATCH_STRIDE):
            for x in range(0, W, PATCH_STRIDE):
                p = img[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
                if np.count_nonzero(p) < 1:            # skip off-breast
                    continue
                mp = m[y:y + PATCH_SIZE, x:x + PATCH_SIZE] if m is not None else None
                recs.append((iid, x, y, int(mp is not None and mp.max() > 0)))
    df = pd.DataFrame(recs, columns=["iid", "x", "y", "pos"])
    if balance:
        pos, neg = df[df.pos == 1], df[df.pos == 0]
        n = min(len(neg), int(len(pos) * NEG_POS_RATIO))
        neg = neg.sample(n=n, random_state=SEED) if n > 0 else neg.iloc[0:0]
        df = pd.concat([pos, neg]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    return df

class PatchDS(Dataset):
    def __init__(self, idx_df, train=False):
        self.df = idx_df.reset_index(drop=True); self.train = train
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]; iid, x, y = r.iid, int(r.x), int(r.y)
        img, m = _img(iid), _mask(iid)
        p = pad(img[y:y + PATCH_SIZE, x:x + PATCH_SIZE], PATCH_SIZE)
        mp = pad((m[y:y + PATCH_SIZE, x:x + PATCH_SIZE] if m is not None
                  else np.zeros((PATCH_SIZE, PATCH_SIZE), np.uint8)), PATCH_SIZE)
        if self.train: p, mp = aug(p, mp)
        return to_tensor(p), torch.from_numpy((mp > 127).astype(np.float32)).unsqueeze(0)

class WholeDS(Dataset):
    def __init__(self, ids, train=False):
        self.ids = list(ids); self.train = train
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        iid = self.ids[i]; img, m = _img(iid), _mask(iid)
        if m is None: m = np.zeros_like(img)
        ir = cv2.resize(img, (WHOLE_SIZE, WHOLE_SIZE), interpolation=cv2.INTER_AREA)
        mr = cv2.resize(m, (WHOLE_SIZE, WHOLE_SIZE), interpolation=cv2.INTER_NEAREST)
        if self.train: ir, mr = aug(ir, mr)
        return to_tensor(ir), torch.from_numpy((mr > 127).astype(np.float32)).unsqueeze(0)


# evaluation
@torch.no_grad()
def eval_patches(model, idx_df):
    ds = PatchDS(idx_df[idx_df.pos == 1], train=False)     # lesion patches
    if len(ds) == 0: return float("nan"), float("nan"), 0
    ld = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    ds_list = []
    model.eval()
    for x, y in ld:
        pr = torch.sigmoid(model(x.to(DEVICE))).cpu().numpy()
        yb = y.numpy()
        for j in range(len(pr)):
            ds_list.append(dice_iou(yb[j, 0], (pr[j, 0] > THRESHOLD).astype(np.float32)))
    a = np.array(ds_list); return a[:, 0].mean(), a[:, 1].mean(), len(a)

@torch.no_grad()
def eval_whole(model, ids):
    model.eval(); res = []
    for iid in ids:
        img, m = _img(iid), _mask(iid)
        if m is None or m.max() == 0: continue             # lesion images only
        H, W = img.shape
        xr = to_tensor(cv2.resize(img, (WHOLE_SIZE, WHOLE_SIZE), interpolation=cv2.INTER_AREA))
        pr = torch.sigmoid(model(xr.unsqueeze(0).to(DEVICE))).cpu().numpy()[0, 0]
        pr = cv2.resize(pr, (W, H), interpolation=cv2.INTER_LINEAR)
        res.append(dice_iou((m > 127).astype(np.float32), (pr > THRESHOLD).astype(np.float32)))
    a = np.array(res); return a[:, 0].mean(), a[:, 1].mean(), len(a)

def load_fold(folds_csv, fold, val_fraction=0.1, seed=42):
    df = pd.read_csv(folds_csv)
    col = f"fold_{fold}"
    if col not in df.columns:
        raise SystemExit(f"{col} not in {folds_csv} (has: {list(df.columns)})")
    names = df["image_name"].astype(str)
    tr = names[df[col] == "train"].tolist()
    va = names[df[col] == "val"].tolist()
    te = names[df[col] == "test"].tolist()
    if not va:                                  
        rng = random.Random(seed + fold)
        k = max(1, int(len(tr) * val_fraction))
        va = rng.sample(tr, k); vs = set(va)
        tr = [i for i in tr if i not in vs]
    return tr, va, te

def run_fold(fold, tr, va, te):
    run_dir = os.path.join(OUTPUT_BASE, f"fold{fold}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"\nfold {fold}: {len(tr)} train, {len(va)} val, {len(te)} test")

    model = MammoClipUNet(BACKBONES[BACKBONE], out_channels=1)
    load_mammo_clip_encoder(model, MAMMO_CKPT)
    model = model.to(DEVICE)

    if MODE == "patch":
        tr_idx = build_patch_index(tr, balance=True)
        va_idx = build_patch_index(va, balance=False)
        te_idx = build_patch_index(te, balance=False)
        print(f"  patches: train {len(tr_idx)} (+{int(tr_idx.pos.sum())}), val {len(va_idx)}, test {len(te_idx)}")
        train_loader = DataLoader(PatchDS(tr_idx, train=True), batch_size=BATCH_SIZE,
                                  shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
        val_eval  = lambda: eval_patches(model, va_idx)
        test_eval = lambda: eval_patches(model, te_idx)
    else:
        train_loader = DataLoader(WholeDS(tr, train=True), batch_size=BATCH_SIZE,
                                  shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
        val_eval  = lambda: eval_whole(model, va)
        test_eval = lambda: eval_whole(model, te)

    criterion = DiceBCELoss()
    enc_params = list(model.encoder.parameters())
    dec_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    optimizer = torch.optim.AdamW([{"params": enc_params, "lr": LR * 0.1},
                                   {"params": dec_params, "lr": LR}], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)

    best, patience, hist = -1.0, 0, {"train_loss": [], "val_dice": []}
    for epoch in range(EPOCHS):
        freeze = epoch < FREEZE_ENCODER_WARMUP
        for p in enc_params: p.requires_grad = not freeze
        model.train(); tl = 0.0
        for x, y in tqdm(train_loader, desc=f"f{fold} Ep{epoch}", leave=False):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y); loss.backward(); optimizer.step()
            tl += loss.item()
        tl /= len(train_loader)
        vd, vi, vn = val_eval()
        hist["train_loss"].append(tl); hist["val_dice"].append(vd)
        scheduler.step(vd)
        frozen = " (encoder frozen)" if freeze else ""
        print(f"  fold {fold} ep {epoch}: loss {tl:.4f}, dice {vd:.4f}, iou {vi:.4f}{frozen}")
        if vd > best + 1e-4:
            best, patience = vd, 0
            torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pth"))
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"  stopping early"); break

    plt.figure(); plt.plot(hist["train_loss"], label="train loss")
    plt.plot(hist["val_dice"], label="val Dice"); plt.legend(); plt.grid(alpha=.3)
    plt.savefig(os.path.join(run_dir, "history.png"), dpi=150); plt.close()

    model.load_state_dict(torch.load(os.path.join(run_dir, "best_model.pth"), map_location=DEVICE))
    td, ti, tn = test_eval()
    summary = {"fold": fold, "mode": MODE, "backbone": BACKBONE,
               "test_dice": td, "test_iou": ti, "n_test": tn, "best_val_dice": best}
    with open(os.path.join(run_dir, "test_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  fold {fold} test: dice={summary['test_dice']:.4f}, iou={summary['test_iou']:.4f}")
    return summary


def main():
    seed_all(SEED)
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    results = []
    for fold in FOLDS:
        tr, va, te = load_fold(FOLDS_CSV, fold, VAL_FRACTION, SEED)
        results.append(run_fold(fold, tr, va, te))
    res = pd.DataFrame(results)
    res.to_csv(os.path.join(OUTPUT_BASE, "cv_results.csv"), index=False)
    print("\ncross-validation summary:")
    print(res.to_string(index=False))
    dice_m, dice_s = res.test_dice.mean(), res.test_dice.std()
    iou_m, iou_s = res.test_iou.mean(), res.test_iou.std()
    print(f"dice {dice_m:.4f} ± {dice_s:.4f}, iou {iou_m:.4f} ± {iou_s:.4f} ({len(res)} folds)")
if __name__ == "__main__":
    main()