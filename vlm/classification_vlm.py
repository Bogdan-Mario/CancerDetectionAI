"""
VLM integration in CLASSIFICATION.

Completes the CNN patch classifier with the VLM's semantic view of each patch:

  (A) ZERO-SHOT VLM          : image-text similarity to clinical class prompts.
  (B) LATE (SCORE) FUSION    : convex combination of CNN softmax and VLM softmax
                               -> a single more robust decision. `alpha` tunable.
  (C) FEATURE FUSION HEAD    : a small MLP over [CNN pre-logit features ||
                               VLM image embedding], trained on top of both
                               frozen encoders -> the strongest combined model.
  (D) AGREEMENT / DIFFICULTY : per-patch CNN-vs-VLM agreement. Disagreement +
                               low confidence flags "hard" cases (the assignment's
                               case-difficulty indicator).

Reuses the existing patch pipeline
(PatchFilterClassificationTrainingAndTesting.py) so labels/patches stay
identical to Tema 1.

Run:
    python vlm/classification_vlm.py            # uses default dirs below
"""
import os, sys, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vlm.vlm_core import VLM                                       # noqa: E402
from vlm.clinical_prompts import CLASS_PROMPTS, CLASS_NAMES, sick_only_prompts  # noqa: E402
from vlm.fusion_ops import late_fusion, sweep_alpha, agreement_analysis  # noqa: E402
# NB: the Tema-1 patch pipeline + sklearn are imported lazily inside main() so
# this module (CNNFeatures / featurize / fusion head) is importable without
# albumentations/matplotlib/sklearn installed.

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- CNN side
class CNNFeatures(nn.Module):
    """timm backbone that returns (logits, pre-logit features)."""
    def __init__(self, backbone="resnet50", num_classes=4, pretrained=True):
        super().__init__()
        import timm
        self.net = timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes)
        self.feat_dim = self.net.num_features

    def forward(self, x):
        f = self.net.forward_head(self.net.forward_features(x), pre_logits=True)
        return self.net.get_classifier()(f), f


def load_cnn(checkpoint=None, backbone="resnet50", num_classes=4):
    m = CNNFeatures(backbone, num_classes, pretrained=checkpoint is None).to(DEVICE)
    if checkpoint and os.path.exists(checkpoint):
        sd = torch.load(checkpoint, map_location=DEVICE)
        sd = sd.get("state_dict", sd)
        # tolerate a bare timm state_dict (keys without the "net." prefix)
        if not any(k.startswith("net.") for k in sd):
            sd = {f"net.{k}": v for k, v in sd.items()}
        missing, unexpected = m.load_state_dict(sd, strict=False)
        print(f"[CNN] loaded {checkpoint} (missing {len(missing)}, unexpected {len(unexpected)})")
    m.eval()
    return m


# ------------------------------------------------------- shared featurization
def _preprocess_cnn(patch, size):
    """grayscale uint8 patch -> normalized CNN tensor (ImageNet stats)."""
    import cv2
    p = cv2.cvtColor(patch, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
    p = cv2.resize(p, (size, size))
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    p = (p - mean) / std
    return torch.from_numpy(p.transpose(2, 0, 1))


@torch.no_grad()
def featurize(cnn, vlm, patches, labels, cnn_size=224, batch=64):
    """
    One pass over patches -> everything the fusion needs, cached:
      cnn_probs (N,C), cnn_feat (N,Fc), vlm_emb (N,D), labels (N,)
    """
    prompts = CLASS_PROMPTS if cnn.net.get_classifier().out_features == 4 else sick_only_prompts()
    keys, bank = vlm.build_text_bank(prompts)
    cnn_probs, cnn_feat, vlm_emb, vlm_probs = [], [], [], []
    for i in range(0, len(patches), batch):
        chunk = patches[i:i + batch]
        x = torch.stack([_preprocess_cnn(p, cnn_size) for p in chunk]).to(DEVICE)
        logits, feat = cnn(x)
        cnn_probs.append(F.softmax(logits, dim=1).cpu().numpy())
        cnn_feat.append(feat.cpu().numpy())
        emb = vlm.encode_image(chunk)                       # (b, D) normalized
        vlm_emb.append(emb.cpu().numpy())
        vlm_probs.append(vlm.similarity(emb, bank, softmax=True).cpu().numpy())
    return {
        "cnn_probs": np.concatenate(cnn_probs),
        "cnn_feat": np.concatenate(cnn_feat),
        "vlm_emb": np.concatenate(vlm_emb),
        "vlm_probs": np.concatenate(vlm_probs),
        "labels": np.asarray(labels),
        "keys": keys,
    }


# (B) late fusion + (D) agreement/difficulty now live in vlm/fusion_ops.py
#     (framework-free, unit-tested). Imported above.


# ---------------------------------------------------- (C) feature fusion head
class FeatureFusionHead(nn.Module):
    """MLP over [CNN features || VLM embedding]."""
    def __init__(self, cnn_dim, vlm_dim, num_classes, hidden=512, p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cnn_dim + vlm_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(True),
            nn.Dropout(p), nn.Linear(hidden, num_classes))

    def forward(self, cnn_feat, vlm_emb):
        return self.net(torch.cat([cnn_feat, vlm_emb], dim=1))


def train_fusion_head(train, val, num_classes, epochs=40, lr=1e-3, wd=1e-4):
    """train/val: dicts from featurize(). Returns (head, val_probs)."""
    cnn_dim, vlm_dim = train["cnn_feat"].shape[1], train["vlm_emb"].shape[1]
    head = FeatureFusionHead(cnn_dim, vlm_dim, num_classes).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)

    def loader(d, bs=256, shuffle=False):
        n = len(d["labels"]); order = np.random.permutation(n) if shuffle else np.arange(n)
        for i in range(0, n, bs):
            idx = order[i:i + bs]
            yield (torch.tensor(d["cnn_feat"][idx]).to(DEVICE),
                   torch.tensor(d["vlm_emb"][idx]).to(DEVICE),
                   torch.tensor(d["labels"][idx]).long().to(DEVICE))

    best, best_state = -1.0, None
    for ep in range(epochs):
        head.train()
        for cf, ve, y in loader(train, shuffle=True):
            opt.zero_grad()
            loss = F.cross_entropy(head(cf, ve), y)
            loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            preds, probs = [], []
            for cf, ve, y in loader(val):
                pr = F.softmax(head(cf, ve), dim=1)
                probs.append(pr.cpu().numpy()); preds.append(pr.argmax(1).cpu().numpy())
        acc = (np.concatenate(preds) == val["labels"]).mean()
        if acc > best:
            best, best_state = acc, {k: v.cpu().clone() for k, v in head.state_dict().items()}
    if best_state:
        head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        vp = [F.softmax(head(cf, ve), dim=1).cpu().numpy() for cf, ve, _ in loader(val)]
    return head, np.concatenate(vp)


# ------------------------------------------- generic trained head (fair baseline)
class _MLPHead(nn.Module):
    def __init__(self, in_dim, num_classes, hidden=512, p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(True),
            nn.Dropout(p), nn.Linear(hidden, num_classes))

    def forward(self, x):
        return self.net(x)


def train_head(Xtr, ytr, Xva, yva, num_classes, epochs=40, lr=1e-3, wd=1e-4, bs=256):
    """
    Train one MLP on a feature matrix -> best val probs (selected by val acc).
    Used to build FAIR baselines: train the SAME head on CNN-features-only, on
    VLM-embedding-only, and on the concatenation, so the gap isolates the VLM's
    marginal contribution (vs. the untrained-head `cnn_only` which is ~chance).
    """
    head = _MLPHead(Xtr.shape[1], num_classes).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    Xva_t = torch.tensor(Xva, dtype=torch.float32).to(DEVICE)
    best_acc, best_probs = -1.0, None
    for _ in range(epochs):
        head.train()
        order = torch.randperm(len(Xtr_t))
        for i in range(0, len(order), bs):
            idx = order[i:i + bs]
            opt.zero_grad()
            loss = F.cross_entropy(head(Xtr_t[idx].to(DEVICE)), ytr_t[idx].to(DEVICE))
            loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            probs = F.softmax(head(Xva_t), dim=1).cpu().numpy()
        acc = (probs.argmax(1) == yva).mean()
        if acc > best_acc:
            best_acc, best_probs = acc, probs
    return best_probs


# --------------------------------------------------------------------- report
def evaluate(name, probs, labels, out_dir):
    from sklearn.metrics import classification_report, confusion_matrix
    preds = probs.argmax(1)
    acc = float((preds == labels).mean())
    rep = classification_report(labels, preds, digits=4, zero_division=0, output_dict=True)
    with open(os.path.join(out_dir, f"report_{name}.json"), "w") as f:
        json.dump({"accuracy": acc, "report": rep,
                   "confusion": confusion_matrix(labels, preds).tolist()}, f, indent=2)
    print(f"  {name:16s} acc = {acc:.4f}")
    return acc


def main():
    from PatchFilterClassificationTrainingAndTesting import (
        extract_patches_from_dataset, seed_everything)
    seed_everything(42)
    IMAGE_DIR = os.environ.get("IMAGE_DIR", "/Normal")
    MASK_DIR = os.environ.get("MASK_DIR", "/Masks")
    CNN_CKPT = os.environ.get("CNN_CKPT", "")          # optional trained CNN
    PATCH_SIZE = int(os.environ.get("PATCH_SIZE", "128"))
    OUT_DIR = os.environ.get("OUT_DIR", "./VLM_Classification_Results")
    os.makedirs(OUT_DIR, exist_ok=True)

    print("== VLM classification integration ==")
    data, num_cls = extract_patches_from_dataset(IMAGE_DIR, MASK_DIR, PATCH_SIZE,
                                                 overlap=True, balance_mode="balanced")
    patches = [d["patch"] for d in data]
    labels = [d["label"] for d in data]
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(patches))
    tr_i, va_i = train_test_split(idx, test_size=0.2, random_state=42, stratify=labels)

    print("loading VLM (BiomedCLIP) + CNN ...")
    vlm = VLM(device=DEVICE)
    cnn = load_cnn(CNN_CKPT or None, backbone="resnet50", num_classes=num_cls)

    print("featurizing (this caches CNN feats + VLM embeddings once)...")
    tr = featurize(cnn, vlm, [patches[i] for i in tr_i], [labels[i] for i in tr_i])
    va = featurize(cnn, vlm, [patches[i] for i in va_i], [labels[i] for i in va_i])

    # (A) raw baselines: untrained-head CNN (~chance unless CNN_CKPT given) + VLM zero-shot
    evaluate("cnn_only_raw", va["cnn_probs"], va["labels"], OUT_DIR)
    evaluate("vlm_zeroshot", va["vlm_probs"], va["labels"], OUT_DIR)

    # (B) late fusion with best alpha chosen on TRAIN
    (best_a, _), curve = sweep_alpha(tr["cnn_probs"], tr["vlm_probs"], tr["labels"])
    print(f"  best alpha (train) = {best_a:.2f}")
    evaluate("late_fusion", late_fusion(va["cnn_probs"], va["vlm_probs"], best_a),
             va["labels"], OUT_DIR)

    # (C) FAIR comparison: SAME trained MLP head on CNN-only / VLM-only / fusion.
    #     acc_fusion - acc_cnn_trained = the VLM's marginal contribution.
    concat_tr = np.concatenate([tr["cnn_feat"], tr["vlm_emb"]], 1)
    concat_va = np.concatenate([va["cnn_feat"], va["vlm_emb"]], 1)
    acc_cnn = evaluate("cnn_trained",
                       train_head(tr["cnn_feat"], tr["labels"], va["cnn_feat"], va["labels"], num_cls),
                       va["labels"], OUT_DIR)
    acc_vlm = evaluate("vlm_trained",
                       train_head(tr["vlm_emb"], tr["labels"], va["vlm_emb"], va["labels"], num_cls),
                       va["labels"], OUT_DIR)
    acc_fus = evaluate("feature_fusion",
                       train_head(concat_tr, tr["labels"], concat_va, va["labels"], num_cls),
                       va["labels"], OUT_DIR)
    print(f"  >> VLM marginal contribution (fusion - cnn_trained) = {acc_fus - acc_cnn:+.4f}")

    # (D) agreement / difficulty
    ag = agreement_analysis(va["cnn_probs"], va["vlm_probs"], va["labels"])
    diff = ag.pop("difficulty")
    np.save(os.path.join(OUT_DIR, "val_difficulty.npy"), diff)
    with open(os.path.join(OUT_DIR, "agreement.json"), "w") as f:
        json.dump(ag, f, indent=2)
    print("  agreement:", json.dumps(ag, indent=2))
    with open(os.path.join(OUT_DIR, "alpha_curve.json"), "w") as f:
        json.dump(curve, f, indent=2)
    print(f"done -> {OUT_DIR}")


if __name__ == "__main__":
    main()
