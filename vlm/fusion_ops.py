"""
Framework-free fusion / analysis ops (numpy + cv2 only, no torch).

These are the numeric primitives shared by the three integration stages. Kept
separate from the torch code so they can be unit-tested without a GPU / model
(see verify_logic.py).
"""
import numpy as np
import cv2


# ============================================================ CLASSIFICATION
def late_fusion(cnn_probs, vlm_probs, alpha=0.6):
    """alpha*CNN + (1-alpha)*VLM, renormalized. Inputs/out: (N, C)."""
    fused = alpha * cnn_probs + (1 - alpha) * vlm_probs
    return fused / fused.sum(1, keepdims=True)


def sweep_alpha(cnn_probs, vlm_probs, labels, grid=None):
    """Return (best=(alpha,acc), curve=[(alpha,acc)...]) maximizing accuracy."""
    if grid is None:
        grid = np.linspace(0, 1, 21)
    curve = []
    for a in grid:
        acc = (late_fusion(cnn_probs, vlm_probs, a).argmax(1) == labels).mean()
        curve.append((float(a), float(acc)))
    return max(curve, key=lambda t: t[1]), curve


def agreement_analysis(cnn_probs, vlm_probs, labels):
    """
    CNN-vs-VLM agreement as a case-difficulty indicator.
    Returns a dict of rates + a per-patch `difficulty` array in [0,1].
    """
    cnn_pred, vlm_pred = cnn_probs.argmax(1), vlm_probs.argmax(1)
    cnn_conf, vlm_conf = cnn_probs.max(1), vlm_probs.max(1)
    agree = cnn_pred == vlm_pred
    difficulty = (~agree).astype(float) * 0.5 + (1 - 0.5 * (cnn_conf + vlm_conf)) * 0.5

    def acc(mask):
        return float((cnn_pred[mask] == labels[mask]).mean()) if mask.any() else float("nan")

    return {
        "agreement_rate": float(agree.mean()),
        "cnn_acc_overall": float((cnn_pred == labels).mean()),
        "cnn_acc_when_agree": acc(agree),
        "cnn_acc_when_disagree": acc(~agree),
        "n_disagree": int((~agree).sum()),
        "n_total": int(len(labels)),
        "difficulty": difficulty,
    }


# =================================================================== DETECTION
def crop_box(img, box, pad=4):
    """Crop [x1,y1,x2,y2] (with padding) from img; None if degenerate."""
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    H, W = img.shape[:2]
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(W, x2 + pad); y2 = min(H, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


def rerank_and_filter(det_scores, vlm_scores, beta=0.5, min_vlm=0.35):
    """
    Geometric-mean fusion of detector + VLM scores, plus a semantic keep-mask.
      fused = det^(1-beta) * vlm^beta ;  keep = vlm_score >= min_vlm
    """
    det = np.clip(np.asarray(det_scores, float), 1e-6, 1.0)
    vlm = np.clip(np.asarray(vlm_scores, float), 1e-6, 1.0)
    fused = det ** (1 - beta) * vlm ** beta
    return fused, np.asarray(vlm_scores, float) >= min_vlm


# ================================================================ SEGMENTATION
def relevance_to_prompts(heat, thr=0.5, max_points=3):
    """
    Relevance heatmap (H,W in [0,1]) -> MedSAM prompts:
      box   : tight bbox around thresholded region [x1,y1,x2,y2]
      points: up to `max_points` component centroids
    Returns dict(box, points, point_labels) or None if nothing salient.
    """
    m = (heat >= thr).astype(np.uint8)
    if m.sum() == 0:
        return None
    ys, xs = np.where(m > 0)
    box = [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]
    n, lab = cv2.connectedComponents(m)
    comps = sorted(range(1, n), key=lambda i: (lab == i).sum(), reverse=True)[:max_points]
    points = []
    for i in comps:
        cy, cx = np.argwhere(lab == i).mean(0)
        points.append([int(cx), int(cy)])
    return {"box": box, "points": points, "point_labels": [1] * len(points)}


def mask_relevance_consistency(mask, heat, thr=0.5):
    """IoU + coverage between a binary mask and the VLM relevance region."""
    m = mask.astype(bool)
    r = heat >= thr
    inter = np.logical_and(m, r).sum()
    union = np.logical_or(m, r).sum()
    iou = float(inter / union) if union > 0 else 1.0
    coverage = float(inter / m.sum()) if m.sum() > 0 else 0.0
    return {"iou": iou, "coverage": coverage, "supported": iou >= 0.1}
