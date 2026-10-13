"""
VLM integration in SEGMENTATION  (functional skeleton).

Two roles, matching the assignment:

  (A) VLM-GUIDED PROMPTS : turn the VLM relevance map into geometric prompts
                           (box and/or point) for box-prompted MedSAM
                           (MedSam2.py). This lets MedSAM segment lesions
                           without a ground-truth box at inference.
  (B) CONSISTENCY CHECK  : compare the VLM relevance region with the produced
                           mask (IoU / coverage). Low agreement flags an
                           unreliable mask -> used to validate / refine or
                           reject segmentation output.

The relevance map itself comes from vlm_core.VLM.relevance_map; here we convert
it into prompts and into a validation signal.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vlm.vlm_core import VLM                                       # noqa: E402
from vlm.clinical_prompts import SEGMENTATION_PROMPTS, DETECTION_POS_PROMPTS, DETECTION_NEG_PROMPTS  # noqa: E402
from vlm.fusion_ops import relevance_to_prompts, mask_relevance_consistency  # noqa: E402


def vlm_prompts_for_patch(vlm, patch, thr=0.5):
    """End-to-end: patch -> MedSAM prompt dict (or None)."""
    heat = vlm.relevance_map(patch, SEGMENTATION_PROMPTS,
                             pos_neg=(DETECTION_POS_PROMPTS, DETECTION_NEG_PROMPTS))
    return relevance_to_prompts(heat, thr=thr), heat


# (B) consistency check `mask_relevance_consistency` now lives in fusion_ops.py.


def build_medsam_box(vlm, patch, thr=0.5):
    """
    Convenience for MedSam2.py inference: returns a box [x1,y1,x2,y2] in patch
    coords derived purely from the VLM (no GT), ready to feed the MedSAM prompt
    encoder. Returns None if the VLM finds nothing salient.
    """
    p, _ = vlm_prompts_for_patch(vlm, patch, thr=thr)
    return None if p is None else p["box"]


def load():
    return VLM()


# ===================================================== evaluation driver ======
# Compares MedSAM segmentation prompted by the GT box vs. a VLM-derived box
# (no ground truth at inference), and reports mask<->relevance consistency.
# Run on the server (needs torch + segment_anything + dataset + MedSAM ckpt):
#   CKPT=/path/medsam_best.pth python vlm/segmentation_vlm.py
def _run_medsam(model, seg, patch, box_patch):
    """Segment one PATCH_SIZE patch with a patch-space box -> binary mask (P,P)."""
    import numpy as np, cv2, torch
    P, S, M = seg.PATCH_SIZE, seg.SAM_SIZE, seg.MASK_SIZE
    img1024 = cv2.resize(patch, (S, S))
    x = torch.from_numpy(np.stack([img1024] * 3, 0).astype(np.float32) / 255.0)[None]
    box1024 = torch.as_tensor(np.asarray(box_patch, np.float32) * (S / P))[None]
    with torch.no_grad():
        logits = model(x.to(seg.DEVICE), box1024.to(seg.DEVICE))
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    return (cv2.resize(prob, (P, P)) > seg.THRESHOLD).astype(np.uint8)


def main():
    import os, json, numpy as np, torch
    sys_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    import sys; sys.path.insert(0, sys_path)
    import MedSam2 as seg
    from segment_anything import sam_model_registry

    for k in ("SOURCE_IMG_DIR", "SOURCE_MASK_DIR", "FOLDS_CSV"):
        v = os.environ.get(k)
        if v:
            setattr(seg, k, v)

    # CKPT = fine-tuned MedSAM wrapper state dict (e.g. medsam_best.pth). It
    # already contains image/prompt/mask weights, so the SAM base checkpoint is
    # optional (SAM_BASE); if absent we build a bare SAM and load CKPT in full.
    ckpt = os.environ.get("CKPT", "")
    sam_base = os.environ.get("SAM_BASE", "") or None
    fold = int(os.environ.get("FOLD", "1"))
    thr = float(os.environ.get("REL_THR", "0.5"))
    out_dir = os.environ.get("OUT_DIR", "./VLM_Segmentation_Results")
    os.makedirs(out_dir, exist_ok=True)

    seg.seed_all(seg.SEED)
    _, _, test_ids = seg.load_fold(fold)
    samples = seg.build_samples(test_ids)          # (iid, px, py, gt_box) lesion patches

    sam = sam_model_registry["vit_b"](checkpoint=sam_base)
    model = seg.MedSAM(sam).to(seg.DEVICE)
    if ckpt and os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=seg.DEVICE))
        print(f"[seg] loaded {ckpt}")
    elif sam_base is None:
        print("[seg] WARNING: no CKPT and no SAM_BASE -> random weights")
    model.eval()
    vlm = VLM()

    def dice(a, b):
        inter = np.logical_and(a > 0, b > 0).sum(); s = (a > 0).sum() + (b > 0).sum()
        return float(2 * inter / s) if s > 0 else 1.0

    d_gt, d_vlm, cons, n_found = [], [], [], 0
    for iid, px, py, gt_box in samples:
        patch = seg.pad(seg._img(iid)[py:py + seg.PATCH_SIZE, px:px + seg.PATCH_SIZE], seg.PATCH_SIZE)
        gmask = seg.pad((seg._mask(iid)[py:py + seg.PATCH_SIZE, px:px + seg.PATCH_SIZE] > 127).astype(np.uint8),
                        seg.PATCH_SIZE)
        # GT-box prompted (upper bound)
        m_gt = _run_medsam(model, seg, patch, gt_box)
        d_gt.append(dice(m_gt, gmask))
        # VLM-box prompted (no GT) + consistency
        prompt, heat = vlm_prompts_for_patch(vlm, patch, thr=thr)
        if prompt is None:
            d_vlm.append(0.0); continue
        n_found += 1
        m_vlm = _run_medsam(model, seg, patch, prompt["box"])
        d_vlm.append(dice(m_vlm, gmask))
        cons.append(mask_relevance_consistency(m_vlm, heat, thr=thr)["iou"])

    summary = {
        "fold": fold, "n_patches": len(samples), "n_vlm_prompted": n_found,
        "dice_gt_box": float(np.mean(d_gt)) if d_gt else 0.0,
        "dice_vlm_box": float(np.mean(d_vlm)) if d_vlm else 0.0,
        "mean_mask_relevance_iou": float(np.mean(cons)) if cons else 0.0,
    }
    with open(os.path.join(out_dir, "segmentation_vlm_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
