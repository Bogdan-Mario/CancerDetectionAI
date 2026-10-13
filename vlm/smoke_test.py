"""
Smoke test of the torch-dependent code paths on SYNTHETIC patches.

Proves the pipeline "engine" actually runs: BiomedCLIP loads and produces
embeddings/similarity/relevance, the real timm CNN produces features, the
feature-fusion head trains, and the detection/segmentation VLM helpers execute.

This is NOT an accuracy test (no mammography data) — it verifies that every
torch code path executes end-to-end and returns well-shaped outputs.

    python vlm/smoke_test.py     # downloads BiomedCLIP (~400MB) on first run

Needs: torch, open_clip_torch, timm, opencv-python(-headless), numpy.
"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vlm.vlm_core import VLM
from vlm.clinical_prompts import CLASS_PROMPTS, DETECTION_POS_PROMPTS, DETECTION_NEG_PROMPTS
import vlm.classification_vlm as C
import vlm.detection_vlm as D
import vlm.segmentation_vlm as S
from vlm.fusion_ops import late_fusion, agreement_analysis

OK, BAD = 0, 0
def check(name, cond, extra=""):
    global OK, BAD
    print((" [PASS] " if cond else " [FAIL] ") + name + ("" if cond else f"   {extra}"))
    OK += bool(cond); BAD += (not cond)

def synth_patch(size=128, kind="tissue", seed=0):
    rng = np.random.default_rng(seed)
    a = rng.integers(40, 120, (size, size), np.uint8)             # background tissue
    if kind == "calc":                                           # bright specks
        for _ in range(25):
            y, x = rng.integers(0, size, 2); a[y:y+2, x:x+2] = 255
    elif kind == "mass":                                        # bright blob
        cv2 = __import__("cv2"); cv2.circle(a, (size//2, size//2), size//5, 220, -1)
    return a

print("== loading VLM (BiomedCLIP) — first run downloads ~400MB ==")
vlm = VLM(device="cpu")
check("VLM loaded", vlm.model is not None)
check("logit_scale > 0", vlm.logit_scale > 0, str(vlm.logit_scale))

patches = [synth_patch(128, k, i) for i, k in enumerate(["tissue","calc","mass","tissue","calc","mass","tissue","mass"])]
labels = np.array([0, 2, 1, 0, 2, 1, 0, 1])

print("== VLM primitives ==")
img_emb = vlm.encode_image(patches[:3])
check("encode_image shape (3,D)", img_emb.ndim == 2 and img_emb.shape[0] == 3, str(tuple(img_emb.shape)))
check("image emb L2-normalized", torch.allclose(img_emb.norm(dim=-1), torch.ones(3), atol=1e-3))
txt = vlm.encode_text(["microcalcifications", "normal tissue"])
check("encode_text shape (2,D)", txt.shape[0] == 2)
keys, bank = vlm.build_text_bank(CLASS_PROMPTS)
check("text bank (4 classes)", bank.shape[0] == 4 and keys == [0,1,2,3], str(keys))
probs, keys2 = vlm.classify_zeroshot(patches, CLASS_PROMPTS)
check("zeroshot probs (8,4) sum~1", probs.shape == (8,4) and np.allclose(probs.sum(1), 1, atol=1e-3), str(probs.shape))
heat = vlm.relevance_map(patches[1], DETECTION_POS_PROMPTS, win=64, stride=32,
                         pos_neg=(DETECTION_POS_PROMPTS, DETECTION_NEG_PROMPTS))
check("relevance_map matches patch HxW", heat.shape == (128,128), str(heat.shape))
check("relevance_map in [0,1]", 0.0 <= heat.min() and heat.max() <= 1.0)

print("== CNN (real timm resnet50) ==")
cnn = C.load_cnn(None, backbone="resnet50", num_classes=4)
x = torch.stack([C._preprocess_cnn(p, 224) for p in patches[:2]]).to(C.DEVICE)  # match CNN device (GPU on Kaggle)
logits, feat = cnn(x)
check("cnn logits (2,4)", tuple(logits.shape) == (2,4), str(tuple(logits.shape)))
check("cnn features (2,2048)", tuple(feat.shape) == (2,2048), str(tuple(feat.shape)))

print("== featurize + fusion (classification) ==")
tr = C.featurize(cnn, vlm, patches[:6], labels[:6])
va = C.featurize(cnn, vlm, patches[6:], labels[6:])
check("featurize keys present", all(k in tr for k in ["cnn_probs","cnn_feat","vlm_emb","vlm_probs"]))
check("cnn_probs (6,4)", tr["cnn_probs"].shape == (6,4), str(tr["cnn_probs"].shape))
check("vlm_emb 2D", tr["vlm_emb"].ndim == 2)
fused = late_fusion(va["cnn_probs"], va["vlm_probs"], 0.6)
check("late_fusion (2,4) rows sum 1", fused.shape == (2,4) and np.allclose(fused.sum(1),1,atol=1e-4))
ag = agreement_analysis(tr["cnn_probs"], tr["vlm_probs"], tr["labels"])
check("agreement_rate in [0,1]", 0 <= ag["agreement_rate"] <= 1, str(ag["agreement_rate"]))
head, vp = C.train_fusion_head(tr, va, num_classes=4, epochs=5)
check("fusion head trained -> val probs (2,4)", vp.shape == (2,4), str(vp.shape))

print("== detection helpers ==")
det_img = synth_patch(256, "calc", 99)
boxes = torch.tensor([[30.,30,70,70],[120.,120,150,150],[200.,10,230,40]])
scores = torch.tensor([0.9, 0.4, 0.7])
vs = D.vlm_box_scores(vlm, det_img, boxes.numpy())
check("vlm_box_scores (3,) in [0,1]", vs.shape == (3,) and vs.min() >= 0 and vs.max() <= 1, str(vs))
fb, fscores = D.apply_vlm_to_detections(vlm, det_img, boxes, scores, beta=0.5, min_vlm=0.0)
check("apply_vlm returns boxes+scores", fb.shape[1] == 4 and len(fscores) == len(fb), str(tuple(fb.shape)))
cand, cs = D.recover_missed_boxes(vlm, det_img, np.zeros((0,4)), patch_size=256, thr=0.99)
check("recover_missed_boxes shape (M,4)", cand.ndim == 2 and cand.shape[1] == 4, str(cand.shape))

print("== segmentation helpers ==")
prompt, heat2 = S.vlm_prompts_for_patch(vlm, patches[1], thr=0.5)
check("vlm_prompts returns box or None", prompt is None or (len(prompt["box"]) == 4))
box = S.build_medsam_box(vlm, patches[1], thr=0.3)
check("build_medsam_box -> 4-list or None", box is None or len(box) == 4, str(box))

print(f"\n==== SMOKE: {OK} passed, {BAD} failed ====")
sys.exit(1 if BAD else 0)
