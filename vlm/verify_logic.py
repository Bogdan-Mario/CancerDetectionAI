"""
Standalone verification of the framework-free fusion/analysis ops.
Runs with only numpy + cv2 (no torch / no model / no dataset needed):

    python vlm/verify_logic.py

Every check has a hand-computed expected value.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vlm.fusion_ops import (late_fusion, sweep_alpha, agreement_analysis,
                            crop_box, rerank_and_filter,
                            relevance_to_prompts, mask_relevance_consistency)

PASS, FAIL = 0, 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1; print(f"  [PASS] {name}")
    else:
        FAIL += 1; print(f"  [FAIL] {name}  {detail}")


print("== late_fusion ==")
cnn = np.array([[0.8, 0.2], [0.3, 0.7]])
vlm = np.array([[0.4, 0.6], [0.1, 0.9]])
f = late_fusion(cnn, vlm, alpha=0.5)
# row0: (0.6,0.4)->norm same; row1: (0.2,0.8)
check("rows sum to 1", np.allclose(f.sum(1), 1.0))
check("alpha=0.5 midpoint", np.allclose(f[0], [0.6, 0.4]) and np.allclose(f[1], [0.2, 0.8]),
      str(f))
check("alpha=1 -> pure CNN", np.allclose(late_fusion(cnn, vlm, 1.0), cnn))
check("alpha=0 -> pure VLM", np.allclose(late_fusion(cnn, vlm, 0.0), vlm))

print("== sweep_alpha ==")
# labels where VLM is right and CNN is wrong on row1 -> higher acc at low alpha
labels = np.array([0, 1])
cnn2 = np.array([[0.9, 0.1], [0.9, 0.1]])   # predicts 0,0  (row1 wrong)
vlm2 = np.array([[0.9, 0.1], [0.1, 0.9]])   # predicts 0,1  (both right)
best, curve = sweep_alpha(cnn2, vlm2, labels)
check("curve length 21", len(curve) == 21)
check("best acc == 1.0", abs(best[1] - 1.0) < 1e-9, str(best))
check("best alpha favors VLM (<1)", best[0] < 1.0, str(best))
check("alpha=1 accuracy is 0.5", abs(dict(curve)[1.0] - 0.5) < 1e-9)

print("== agreement_analysis ==")
# 4 samples: agree+correct, agree+wrong, disagree, disagree
cnnp = np.array([[0.9, 0.1], [0.9, 0.1], [0.8, 0.2], [0.6, 0.4]])
vlmp = np.array([[0.8, 0.2], [0.7, 0.3], [0.2, 0.8], [0.3, 0.7]])
lbl = np.array([0, 1, 0, 1])       # cnn preds: 0,0,0,0 ; vlm preds: 0,0,1,1
ag = agreement_analysis(cnnp, vlmp, lbl)
check("agreement_rate 0.5", abs(ag["agreement_rate"] - 0.5) < 1e-9, str(ag["agreement_rate"]))
check("n_disagree 2", ag["n_disagree"] == 2)
check("cnn_acc_overall 0.5", abs(ag["cnn_acc_overall"] - 0.5) < 1e-9)
check("acc_when_agree 0.5", abs(ag["cnn_acc_when_agree"] - 0.5) < 1e-9)   # s0 correct, s1 wrong
check("acc_when_disagree 0.5", abs(ag["cnn_acc_when_disagree"] - 0.5) < 1e-9)  # s2 correct,s3 wrong
check("difficulty len 4", len(ag["difficulty"]) == 4)
check("disagree harder than agree", ag["difficulty"][2] > ag["difficulty"][0],
      str(ag["difficulty"]))
check("difficulty in [0,1]", ag["difficulty"].min() >= 0 and ag["difficulty"].max() <= 1)

print("== crop_box ==")
img = np.arange(100, dtype=np.uint8).reshape(10, 10)
c = crop_box(img, [3, 3, 6, 6], pad=0)
check("crop shape 3x3", c.shape == (3, 3), str(c.shape))
check("crop clamps to image", crop_box(img, [-5, -5, 3, 3], pad=0).shape == (3, 3))
check("degenerate -> None", crop_box(img, [5, 5, 5, 5], pad=0) is None)

print("== rerank_and_filter ==")
det = np.array([0.9, 0.9, 0.2])
vs = np.array([0.9, 0.1, 0.8])
fused, keep = rerank_and_filter(det, vs, beta=0.5, min_vlm=0.35)
check("geom mean box0", abs(fused[0] - np.sqrt(0.9 * 0.9)) < 1e-6, str(fused[0]))
check("geom mean box1", abs(fused[1] - np.sqrt(0.9 * 0.1)) < 1e-6, str(fused[1]))
check("keep drops low-vlm box1", keep.tolist() == [True, False, True], str(keep))
check("fused box1 < box0 (FP demoted)", fused[1] < fused[0])

print("== relevance_to_prompts ==")
heat = np.zeros((20, 20), np.float32)
heat[5:9, 5:9] = 0.9          # one blob
heat[14:16, 14:16] = 0.8      # second smaller blob
p = relevance_to_prompts(heat, thr=0.5, max_points=3)
check("box wraps both blobs", p["box"] == [5, 5, 16, 16], str(p["box"]))
check("two components found", len(p["points"]) == 2, str(p["points"]))
check("largest blob first", p["points"][0][0] < 10 and p["points"][0][1] < 10, str(p["points"]))
check("point_labels all positive", set(p["point_labels"]) == {1})
check("empty heat -> None", relevance_to_prompts(np.zeros((5, 5)), thr=0.5) is None)

print("== mask_relevance_consistency ==")
mask = np.zeros((10, 10), np.uint8); mask[2:6, 2:6] = 1     # 16 px
h = np.zeros((10, 10), np.float32); h[2:6, 2:6] = 1.0       # identical region
r = mask_relevance_consistency(mask, h, thr=0.5)
check("iou==1 when identical", abs(r["iou"] - 1.0) < 1e-9, str(r))
check("coverage==1 when identical", abs(r["coverage"] - 1.0) < 1e-9)
h2 = np.zeros((10, 10), np.float32); h2[4:8, 4:8] = 1.0     # overlap 2x2=4
r2 = mask_relevance_consistency(mask, h2, thr=0.5)
# inter=4, union=16+16-4=28 -> iou=4/28
check("iou partial overlap 4/28", abs(r2["iou"] - 4 / 28) < 1e-6, str(r2["iou"]))
check("coverage partial 4/16", abs(r2["coverage"] - 0.25) < 1e-9, str(r2["coverage"]))
h3 = np.zeros((10, 10), np.float32)                          # no relevance
r3 = mask_relevance_consistency(mask, h3, thr=0.5)
check("unsupported when no overlap", r3["supported"] is False and r3["iou"] == 0.0)

print(f"\n==== {PASS} passed, {FAIL} failed ====")
sys.exit(1 if FAIL else 0)
