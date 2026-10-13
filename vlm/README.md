# Tema 2 — Integrarea unui Vision-Language Model (VLM) în clasificare, detecție și segmentare

Acest pachet (`vlm/`) adaugă un strat semantic **VLM** peste pipeline-ul existent
de mamografie (Tema 1), fără a re-antrena modelele vizuale de bază. VLM-ul
furnizează cele trei primitive cerute de temă — **scoruri de similaritate
imagine-text**, **embedding-uri** într-un spațiu comun și **hărți de relevanță** —
care sunt apoi folosite diferit în fiecare etapă.

## VLM ales

**BiomedCLIP** (`microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`):
encoder de imagine ViT-B/16 + encoder de text PubMedBERT, proiectați într-un
spațiu de embedding comun. Este exact modelul deja folosit în
`PatchFilterClassificationTrainingAndTesting.get_model("medclip")`, deci
pipeline-ul rămâne consistent. Se încarcă prin `open_clip`.

> Mammo-CLIP (greutățile `b5-*.tar` din proiect) poate fi adăugat ulterior ca al
> doilea semnal semantic specific domeniului — arhitectura din `vlm_core.py` este
> agnostică la modelul concret.

## Structura pachetului

| Fișier | Rol |
|---|---|
| `clinical_prompts.py` | Descrierile clinice (text) per clasă + prompturi pentru detecție/segmentare. Prompt-ensembling. |
| `vlm_core.py` | Wrapper BiomedCLIP: `encode_image`, `encode_text`, `build_text_bank`, `similarity`, `classify_zeroshot`, `relevance_map`. |
| `fusion_ops.py` | Logica numerică pură (numpy/cv2, fără torch): fuziune, acord, reranking, prompturi, consistență. Testabilă izolat. |
| `classification_vlm.py` | **Integrare completă în clasificare** (zero-shot, fuziune scor, cap de fuziune feature, acord/dificultate). |
| `detection_vlm.py` | Detecție: validare + reranking boxuri, eliminare FP, recuperare FN via relevanță + **driver FROC înainte/după**. |
| `segmentation_vlm.py` | Segmentare: prompturi VLM pentru MedSAM + consistență mască↔relevanță + **driver Dice GT-box vs VLM-box**. |
| `verify_logic.py` | Teste standalone (numpy/cv2) pentru `fusion_ops` — 33 verificări cu valori așteptate calculate manual. |
| `smoke_test.py` | Rulează BiomedCLIP + ResNet50 reale pe patch-uri sintetice — 22 verificări că motorul torch se execută end-to-end. |

## 1. Clasificare (complet implementat)

Pentru fiecare patch, VLM-ul estimează similaritatea cu setul de descrieri
clinice (`țesut normal`, `masă`, `microcalcificări suspecte`, `masă+calcificări`).
Se combină cu clasificatorul CNN pe trei niveluri, tot mai puternice:

- **(A) Zero-shot VLM** — softmax pe similaritatea cosinus imagine↔prototip-text.
- **(B) Fuziune la nivel de scor** — `alpha·CNN + (1−alpha)·VLM`, cu `alpha`
  ales pe train (`sweep_alpha`). Decizie finală mai robustă.
- **(C) Cap de fuziune la nivel de feature** — MLP antrenat peste
  `[feature-uri CNN pre-logit ‖ embedding imagine VLM]`, ambele encodere înghețate.
- **(D) Acord CNN↔VLM ca indicator de dificultate** — `agreement_analysis`
  raportează rata de acord și acuratețea separat pe cazurile în acord vs.
  dezacord; scorul `difficulty` per-patch marchează cazurile grele (dezacord +
  încredere scăzută) pentru triaj.

Rulare:
```bash
pip install open_clip_torch torch timm opencv-python albumentations scikit-learn
IMAGE_DIR=/Normal MASK_DIR=/Masks PATCH_SIZE=128 \
  python vlm/classification_vlm.py
```
Produce în `VLM_Classification_Results/`: rapoarte per metodă
(`cnn_only`, `vlm_zeroshot`, `late_fusion`, `feature_fusion`), `agreement.json`,
`alpha_curve.json`, `val_difficulty.npy`.

> `CNN_CKPT=/cale/clasificator.pth` folosește un CNN deja antrenat (Tema 1);
> altfel se folosește un ResNet50 pretrained ImageNet.

## 2. Detecție

Operează pe **ieșirile** RetinaNet (`detectionRetinaNet.predict_image`), fără
re-antrenare:

- **Validare + reranking** (`apply_vlm_to_detections`): fiecare box e decupat și
  scorat de VLM (`microcalcificare reală` vs `normal/artefact`); scorul se
  fuzionează geometric cu scorul detectorului, iar boxurile inconsistente
  (`vlm_score < min_vlm`) sunt eliminate → mai puține fals pozitive.
- **Recuperare FN** (`recover_missed_boxes`): pe patch-urile lăsate goale de
  detector se construiește o hartă de relevanță VLM; vârfurile devin boxuri
  candidate care recuperează leziuni omise (ramura „empty patch → detection” din
  figura pipeline-ului).

Driver de evaluare (FROC înainte vs. după VLM, reutilizează `froc_boxes`):
```bash
CKPT=/path/retina_detection_best_model.pth FOLD=1 python vlm/detection_vlm.py
```
→ `VLM_Detection_Results/detection_vlm_summary.json` (sensibilitate @FP baseline vs VLM).

## 3. Segmentare

- **Prompturi ghidate de VLM** (`vlm_prompts_for_patch`, `build_medsam_box`):
  harta de relevanță → box/puncte pentru MedSAM (`MedSam2.py`), permițând
  segmentarea fără box ground-truth la inferență.
- **Consistență mască↔relevanță** (`mask_relevance_consistency`): IoU/acoperire
  între regiunea evidențiată de VLM și masca produsă; valori mici semnalează
  măști nesusținute semantic → validare/rafinare.

Driver de evaluare (Dice cu box GT vs. box VLM, reutilizează `MedSam2`):
```bash
CKPT=/path/medsam_best.pth FOLD=1 python vlm/segmentation_vlm.py
```
→ `VLM_Segmentation_Results/segmentation_vlm_summary.json`
(`dice_gt_box`, `dice_vlm_box`, `mean_mask_relevance_iou`).

## Verificare

Două niveluri, ambele rulate real:

1. **Logica numerică** (fără torch/dataset) — `fusion_ops.py`:
   ```bash
   python vlm/verify_logic.py     # 33 passed, 0 failed
   ```
2. **Motorul torch pe patch-uri sintetice** — `smoke_test.py` încarcă efectiv
   BiomedCLIP + ResNet50 și rulează toate căile de cod (embeddings, similaritate,
   relevance map, featurizare, antrenarea capului de fuziune, helper-ele de
   detecție/segmentare):
   ```bash
   pip install torch open_clip_torch transformers timm opencv-python-headless
   python vlm/smoke_test.py       # 22 passed, 0 failed (descarcă BiomedCLIP ~400MB)
   ```
   Nu e test de acuratețe (date sintetice) — dovedește că pipeline-ul se execută
   end-to-end și întoarce ieșiri de forma corectă. Metricile reale (acuratețe,
   FROC, Dice) se obțin pe mediul cu GPU + datasetul de mamografie.

> Dependență descoperită la smoke test: **`transformers`** e necesar pentru text
> encoder-ul BiomedCLIP (PubMedBERT) — inclus în notebook și în comenzile de rulare.

## Note

- Codul este scris pentru mediul de antrenare (GPU + dataset în `/Normal`,
  `/Masks`), la fel ca scripturile Tema 1. Local nu există `torch`/dataset, deci
  se rulează pe server/Colab.
- Harta de relevanță e implementată robust prin fereastră glisantă multi-scală pe
  similaritate imagine-text (agnostică la arhitectură); poate fi înlocuită
  ulterior cu o variantă bazată pe token-uri ViT / gradienți.
