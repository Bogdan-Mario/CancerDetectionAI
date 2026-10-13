# Tutorial: cum rulez Tema 2 pe Kaggle (pas cu pas, ce apeși unde)

Scop: să obții metricile reale (acuratețe / FROC / Dice) folosind datasetul tău de
pe Kaggle + GPU-ul gratuit Kaggle. Durează ~15 min de pregătire + timp de rulare.

---

## Partea A — Urcă CODUL pe Kaggle (o singură dată)

Codul (pachetul `vlm/` + scripturile Tema 1) trebuie să ajungă pe Kaggle ca
„Dataset”.

1. Pe calculatorul tău, fă un **zip** din folderul `BreastCancer`
   (`C:\Users\rauce\Desktop\practica\BreastCancer`). Click-dreapta → *Send to* →
   *Compressed (zipped) folder*. Rezultă `BreastCancer.zip`.
2. Intră pe **https://www.kaggle.com** → autentifică-te.
3. Stânga sus: **Create** (butonul `+`) → **New Dataset**.
4. Trage `BreastCancer.zip` în fereastră (Kaggle îl dezarhivează automat).
5. Sus dă-i un titlu, ex: **`breastcancer-vlm-code`** → **Create**.
6. Așteaptă să termine procesarea (câteva secunde, e cod, nu date).

> Alternativ, dacă ai deja repo-ul pe GitHub, poți sări peste asta și folosi
> `!git clone` în notebook — dar zip-ul e cel mai simplu.

---

## Partea B — Găsește / notează datasetul de mamografie

1. Deschide pagina datasetului tău de mamografie pe Kaggle (cel pe care îl ai deja).
2. Notează-i numele (slug-ul din URL, ex: `.../datasets/user/mammo-patches`).
   Îl vei atașa la notebook în Partea C.

---

## Partea C — Creează Notebook-ul și atașează tot

1. **Create** (`+`) → **New Notebook**.
2. În dreapta, panoul **Settings** (dacă nu-l vezi, click pe iconița `⚙`/`≡`):
   - **Accelerator** → alege **GPU** (T4 x2 sau P100).
   - **Internet** → comută pe **On**  ← *obligatoriu*, altfel pip și BiomedCLIP
     nu se descarcă.
3. Tot în dreapta: **Add Input** (sau `+ Add Data`):
   - caută și adaugă **datasetul de mamografie**;
   - caută și adaugă **`breastcancer-vlm-code`** (codul urcat la Partea A);
   - dacă ai și checkpoint-urile `.pth` urcate ca dataset, adaugă-le și pe ele.
4. Încarcă notebook-ul meu: sus **File → Import Notebook** →
   încarcă `vlm/Tema2_VLM_Kaggle.ipynb` (din folderul BreastCancer).
   (Sau: copiază conținutul celulelor manual într-un notebook nou.)

---

## Partea D — Rulează (ordinea celulelor)

Apasă pe fiecare celulă și **Shift+Enter**, de sus în jos:

1. **Celula 0** — verifică GPU și listează ce ai atașat în `/kaggle/input`.
2. **Celula 1** — instalează dependențele (~2-3 min). Așteaptă „gata”.
3. **Celula 2** — găsește automat codul și îl copiază în zona de lucru.
   Dacă dă eroare „Nu am găsit codul”, înseamnă că nu ai atașat
   `breastcancer-vlm-code` la Input.
4. **Celula 3** — *descoperă structura datasetului*: îți listează folderele cu
   imagini `.png` și ghicește `Mass_Only` / `Calcification_Only` / `folds.csv`.
   **Uită-te la ce printează.**
5. **CONFIG (celula cu căile)** — completează `IMG_DIR`, `MASK_DIR`, `CALC_MASK`
   cu exact căile listate la pasul 4. Rulează → trebuie să vezi `OK` peste tot.
6. **Celula 4** — testele de logică (confirmare rapidă că totul e legat).
7. **Etapa 1 — Clasificare**: rulează cele 2 celule. Vei vedea acuratețea pentru
   `cnn_only`, `vlm_zeroshot`, `late_fusion`, `feature_fusion` + acordul CNN↔VLM.
   → *Aceasta e prima ta cifră reală.* Nu are nevoie de niciun checkpoint.
8. **Etapa 2 — Detecție** și **Etapa 3 — Segmentare**: rulează doar dacă ai pus
   `RETINA_CKPT`, respectiv `MEDSAM_CKPT` în CONFIG (și `folds.csv` există).

---

## Ce urmăresc în rezultate

- **Clasificare**: `late_fusion` / `feature_fusion` ar trebui ≥ `cnn_only`
  (VLM-ul ajută). `agreement_rate` mic + acuratețe mai mică pe „disagree” =
  indicatorul de dificultate funcționează.
- **Detecție**: `vlm` sensibilitate @FP ≥ `baseline` = mai puține fals pozitive /
  leziuni recuperate.
- **Segmentare**: `dice_vlm_box` apropiat de `dice_gt_box` = prompturile VLM
  înlocuiesc bine box-ul ground-truth; `mean_mask_relevance_iou` mare = măști
  susținute semantic.

Toate se salvează în `/kaggle/working/VLM_*_Results/` — le poți descărca din
tab-ul **Output** al notebook-ului.

---

## Dacă se blochează ceva

- **„Nu am găsit codul”** → n-ai atașat `breastcancer-vlm-code` la Input.
- **`LIPSA` la CONFIG** → căile din CONFIG nu se potrivesc; copiază exact ce a
  listat celula 3.
- **Eroare la BiomedCLIP / pip** → Internet e Off în Settings.
- **`folds.csv` NU** → detecția/segmentarea au nevoie de el; asigură-te că e în
  folderul de cod sau în dataset, și pune calea în `FOLDS_CSV`.
- **Numele imaginilor din `folds.csv` nu se potrivesc cu datasetul** → spune-mi
  cum arată `folds.csv` și structura reală, și ajustez codul.

Dacă ceva pică, copiază-mi mie output-ul celulei 3 (structura) + mesajul de
eroare și îl rezolv.
