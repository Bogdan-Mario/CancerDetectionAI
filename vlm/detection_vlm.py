"""
VLM integration in DETECTION  (functional skeleton).

Wraps the RetinaNet detector (detectionRetinaNet.py) with a VLM semantic gate:

  (A) VALIDATE + RERANK : for every predicted box, crop the region and score it
                          with the VLM as "real microcalcification" vs
                          "normal/artifact". Fuse VLM score with the detector
                          score -> reranked, FP-suppressed boxes.
  (B) FN RECOVERY       : build a VLM relevance map over patches the detector
                          left empty; peaks become candidate boxes that can
                          rescue missed lesions (matches the pipeline's
                          "empty patch -> detection" FN-recovery branch).

These operate on detector *outputs*, so no retraining is needed — plug them
after RetinaNet.predict_image(). Evaluation reuses froc_boxes / eval_full.
"""
import os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vlm.vlm_core import VLM                                       # noqa: E402
from vlm.clinical_prompts import DETECTION_POS_PROMPTS, DETECTION_NEG_PROMPTS  # noqa: E402
from vlm.fusion_ops import crop_box as _crop, rerank_and_filter    # noqa: E402


# ------------------------------------------------------- (A) validate + rerank
@torch.no_grad()
def vlm_box_scores(vlm, img, boxes):
    """Per-box P(real microcalcification) from the VLM (pos-vs-neg softmax)."""
    if len(boxes) == 0:
        return np.zeros((0,), np.float32)
    pos = vlm.encode_text(DETECTION_POS_PROMPTS).mean(0, keepdim=True)
    neg = vlm.encode_text(DETECTION_NEG_PROMPTS).mean(0, keepdim=True)
    bank = F.normalize(torch.cat([pos, neg], 0), dim=-1)          # (2, D)
    crops = [_crop(img, b) for b in np.asarray(boxes)]
    ok = [c for c in crops if c is not None and min(c.shape[:2]) >= 4]
    scores = np.zeros((len(boxes),), np.float32)
    if not ok:
        return scores
    emb = vlm.encode_image(ok)
    p = vlm.similarity(emb, bank, softmax=True)[:, 0].cpu().numpy()
    j = 0
    for i, c in enumerate(crops):
        if c is not None and min(c.shape[:2]) >= 4:
            scores[i] = p[j]; j += 1
    return scores


def apply_vlm_to_detections(vlm, img, boxes, scores, beta=0.5, min_vlm=0.35):
    """Full (A) step: returns filtered (boxes, fused_scores) as torch tensors."""
    boxes_np = boxes.cpu().numpy() if torch.is_tensor(boxes) else np.asarray(boxes)
    scores_np = scores.cpu().numpy() if torch.is_tensor(scores) else np.asarray(scores)
    vlm_s = vlm_box_scores(vlm, img, boxes_np)
    fused, keep = rerank_and_filter(scores_np, vlm_s, beta, min_vlm)
    return (torch.as_tensor(boxes_np[keep], dtype=torch.float32),
            torch.as_tensor(fused[keep], dtype=torch.float32))


# --------------------------------------------------------- (B) FN recovery
def recover_missed_boxes(vlm, img, occupied_boxes, patch_size=256,
                         win=64, stride=32, thr=0.6, box=48):
    """
    Scan patches with no detector box; use the VLM relevance map to propose
    candidate boxes on high-activation peaks (missed-lesion rescue).

    occupied_boxes: array (K,4) of already-detected boxes (to skip covered area).
    Returns (cand_boxes (M,4), cand_scores (M,)).
    """
    H, W = img.shape[:2]
    occ = np.zeros((H, W), bool)
    for x1, y1, x2, y2 in np.asarray(occupied_boxes).reshape(-1, 4).astype(int):
        occ[max(0, y1):y2, max(0, x1):x2] = True

    cand, cs = [], []
    for py in range(0, H, patch_size):
        for px in range(0, W, patch_size):
            patch = img[py:py + patch_size, px:px + patch_size]
            if patch.size == 0 or np.count_nonzero(patch) < 1:
                continue
            if occ[py:py + patch_size, px:px + patch_size].mean() > 0.5:
                continue                                    # already covered
            heat = vlm.relevance_map(patch, DETECTION_POS_PROMPTS, win=win, stride=stride,
                                     pos_neg=(DETECTION_POS_PROMPTS, DETECTION_NEG_PROMPTS))
            ys, xs = np.where(heat >= thr)
            if len(xs) == 0:
                continue
            cy, cx = int(ys.mean()), int(xs.mean())         # peak centroid
            gx, gy = px + cx, py + cy
            b = [gx - box // 2, gy - box // 2, gx + box // 2, gy + box // 2]
            cand.append(b); cs.append(float(heat[cy, cx]))
    return (np.array(cand, np.float32).reshape(-1, 4), np.array(cs, np.float32))


# --------------------------------------------------------------- convenience
def load():
    return VLM()


# ===================================================== evaluation driver ======
# Chains VLM (A)+(B) onto the trained RetinaNet and reports FROC/mAP before vs
# after. Run on the training server (needs torch + dataset + a trained ckpt):
#   CKPT=/path/retina_detection_best_model.pth python vlm/detection_vlm.py
def main():
    import os, json
    import torch
    sys_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    import sys; sys.path.insert(0, sys_path)
    import detectionRetinaNet as det

    # let the notebook/env point the Tema-1 module at the real data + init
    for k in ("SOURCE_IMG_DIR", "SOURCE_MASK_DIR", "FOLDS_CSV", "MAMMO_CKPT", "ENCODER_INIT"):
        v = os.environ.get(k)
        if v:
            setattr(det, k, v)

    ckpt = os.environ.get("CKPT", "")
    fold = int(os.environ.get("FOLD", "1"))
    beta = float(os.environ.get("BETA", "0.5"))
    min_vlm = float(os.environ.get("MIN_VLM", "0.35"))
    do_recover = os.environ.get("RECOVER", "1") == "1"
    out_dir = os.environ.get("OUT_DIR", "./VLM_Detection_Results")
    os.makedirs(out_dir, exist_ok=True)

    det.seed_all(det.SEED)
    _, _, test_ids = det.load_fold(fold)
    model = det.build_model()
    if ckpt and os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=det.DEVICE))
        print(f"[det] loaded {ckpt}")
    model.eval()
    vlm = VLM()

    base_pi, vlm_pi = [], []          # per-image (gt, boxes, scores) before/after
    for iid in test_ids:
        img = det._img(iid)
        boxes, scores = det.predict_image(model, iid)
        gt = torch.as_tensor(det.image_boxes(iid), dtype=torch.float32).reshape(-1, 4)
        base_pi.append((gt, boxes, scores))

        # (A)+(B) semantic rerank + FP filter
        vb, vs = apply_vlm_to_detections(vlm, img, boxes, scores, beta, min_vlm)
        # (FN recovery) add VLM candidate boxes on empty regions
        if do_recover:
            cand, cscore = recover_missed_boxes(vlm, img, vb.numpy())
            if len(cand):
                vb = torch.cat([vb, torch.as_tensor(cand)], 0)
                vs = torch.cat([vs, torch.as_tensor(cscore)], 0)
        vlm_pi.append((gt, vb, vs))

    base_curve, base_sens = det.froc_boxes(base_pi)
    vlm_curve, vlm_sens = det.froc_boxes(vlm_pi)
    summary = {"fold": fold, "beta": beta, "min_vlm": min_vlm, "recover": do_recover,
               "baseline": base_sens, "vlm": vlm_sens, "n_images": len(test_ids)}
    with open(os.path.join(out_dir, "detection_vlm_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("baseline FROC:", base_sens)
    print("VLM      FROC:", vlm_sens)
    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
