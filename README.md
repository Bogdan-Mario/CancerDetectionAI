# BreastCancer

Code for finding and outlining breast cancer signs (masses and tiny calcifications) in mammograms. There are two parts:

1. **The base pipeline**: three models that each do one job.
   - Patch classification (ResNet50 or BiomedCLIP): 0 = healthy, 1 = mass, 2 = calcification, 3 = both.
   - Detection of microcalcifications (RetinaNet with an EfficientNet-B5 backbone).
   - Segmentation (U-Net variants, plus MedSAM when you give it a box).
2. **The `vlm/` folder**: a vision-language model (BiomedCLIP) added on top of the base pipeline. It checks and re-scores what the other models say. We don't retrain the base models for this.

## What's in here

```
BreastCancer/
├── PatchFilterClassificationTrainingAndTesting.py   # patch classification
├── detectionRetinaNet.py                            # detection + FROC / mAP
├── MedSam2.py                                       # MedSAM, prompted with a box
├── mammo_clip_unet*.py, train_mammoclip_*.py        # U-Net segmentation + training
├── download_Mammo-CLIP_weights.py                   # grabs the Mammo-CLIP weights
├── folds.csv                                        # train/val/test split (fold_1..fold_5)
└── vlm/                                             # the VLM part (see below)
```

Inside `vlm/`:

- `vlm_core.py`: wraps BiomedCLIP (image and text embeddings, similarity, relevance maps).
- `clinical_prompts.py`: the text descriptions we compare images against ("a bright cluster of microcalcifications", etc.).
- `fusion_ops.py`: the plain numpy math (merging scores, reranking boxes). It needs no torch, so it's easy to test.
- `classification_vlm.py`, `detection_vlm.py`, `segmentation_vlm.py`: the VLM added to each task.
- `verify_logic.py` and `smoke_test.py`: tests. The first checks the math without torch. The second runs the real models on fake patches to check nothing crashes.
- `Tema2_Final_Kaggle.ipynb`: the notebook that runs everything (classification, then detection, then segmentation).
- `build_notebook_*.py`: scripts that generate the notebooks. **Edit these, not the `.ipynb` files**, then re-run them. They check every cell's syntax so a broken cell doesn't reach Kaggle.
- `TUTORIAL_Kaggle.md` and `vlm/README.md`: step-by-step guides (in Romanian).

## What the VLM actually did

Results from the real runs on Kaggle:

**Classification** (42k patches at 128 px, same small head on frozen features for a fair comparison)

| Method | Accuracy |
|---|---|
| CNN only (trained head) | 0.8245 |
| VLM only (trained head) | 0.8253 |
| CNN + VLM features together | **0.8465** |

The VLM adds about +2.2%. It's small, but it showed up every time.

**Detection** (RetinaNet boxes, VLM re-scores each one; checked on all 5 folds)

- With few false positives allowed (0.5 to 1 per image), the VLM helps in all 5 folds. For example, sensitivity at 1 FP went from about 0.51 to 0.55 on average.
- With many false positives allowed (4 per image), it hurts in all 5 folds (0.79 down to 0.73).
- So use it as a strict filter. Don't use it everywhere.
- One bug mattered a lot: cropping the box too tightly gave the VLM nothing to look at. Adding context around the box flipped its score from worse than random (AUC 0.33) to useful (AUC 0.70).

**Segmentation** (MedSAM needs a box, so the question is where the box comes from)

| Box comes from | Dice |
|---|---|
| Ground truth (upper limit) | 0.76 |
| RetinaNet box, reranked by the VLM | **0.52** |
| VLM relevance map directly | 0.46 |

The VLM alone can't pinpoint microcalcifications. They're only 5 to 20 pixels wide. It does fine on larger lesions, though.

**In short:** the VLM helps with the rough, "what is this" questions. It's weak at "exactly where".

## Running it

The code was built for a GPU machine with the dataset. We ran everything on Kaggle (T4 GPU). Your laptop is fine for the quick checks:

```bash
pip install numpy opencv-python
python vlm/verify_logic.py        # math tests, no torch needed
```

For the real thing:

1. Upload the code, the dataset, and the checkpoints to Kaggle (`TUTORIAL_Kaggle.md` walks through it).
2. Open `vlm/Tema2_Final_Kaggle.ipynb`.
3. Set `SMOKE = True` for a quick check (about 5 min), then `False` for the full run (about 1.5 h).
4. Use **Save Version, then Save & Run All**. Closing the tab kills an interactive session.

You'll need `open_clip_torch`, `transformers`, `timm`, `torch`, and `opencv-python`. `transformers` is easy to forget, but BiomedCLIP's text encoder won't load without it.

## Not in the repo

Model weights (`.pth`) are too big for GitHub, so they're not here. You need:

- `retina_detection_best_model.pth` for detection
- `medsam_best.pth` for MedSAM (it's a full checkpoint, no separate SAM download needed)

The dataset isn't included either (InBreast: `Normal/` and `Masks/`).

## Handy to know

- `detectionRetinaNet.py` has the dataset paths hardcoded. The VLM scripts override them with environment variables.
- Patches are classified on their own, so `folds.csv` is only used for detection and segmentation.
- If you print long output on Kaggle, use `python -u` and `flush=True`, or it looks frozen.
