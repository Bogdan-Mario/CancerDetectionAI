"""
Construieste notebook-ul COMPLET al Temei 2 (Etapa 1 + 2 + 3) intr-un singur
fisier: Tema2_Final_Kaggle.ipynb.

Nu re-tasteaza cod cunoscut-bun: preia celulele de detectie/segmentare/ablatie/
sumar din `Tema2_Etape23_Kaggle.ipynb` (deja validat) si celula de clasificare
inline din `Tema2_VLM_Kaggle.ipynb`. Singurele bucati scrise/modificate aici:
intro, CONFIG (adauga PATCH_SIZE/MASK_DIR/N_CLS), clasificarea (adauga comutatorul
SMOKE + progres), si sumarul final (afiseaza toate 3 etapele).

Ruleaza local (fara torch): `python build_notebook_final.py`. Valideaza fiecare
celula de cod cu ast.parse.
"""
import json, io, os, ast

VLM_DIR = r"C:\Users\rauce\Desktop\practica\BreastCancer\vlm"
ET23 = os.path.join(VLM_DIR, "Tema2_Etape23_Kaggle.ipynb")
CLS  = os.path.join(VLM_DIR, "Tema2_VLM_Kaggle.ipynb")
OUT  = os.path.join(VLM_DIR, "Tema2_Final_Kaggle.ipynb")


def load_cells(p):
    return json.load(io.open(p, encoding="utf-8"))["cells"]


def src(cell):
    s = cell["source"]
    return s if isinstance(s, str) else "".join(s)


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s}


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": s}


def must_replace(text, old, new):
    """String-replace care esueaza ZGOMOTOS daca ancora nu se potriveste."""
    assert old in text, f"Ancora lipseste: {old!r}"
    return text.replace(old, new)


e = load_cells(ET23)   # 0 intro,1 gpu,2 deps,3 copy,4 CONFIG,5 seg-md,6 seg,
                       # 7 seg-viz-md,8 seg-viz,9 seg-plots-md,10 seg-plots,
                       # 11 det-md,12 det,13 det-plots-md,14 det-plots,
                       # 15 det-viz-md,16 det-viz,17 abl-md,18 abl,
                       # 19 abl-plot-md,20 abl-plot,21 gen-md,22 gen,
                       # 23 gen-plot-md,24 gen-plot,25 report-md,26 report,27 sumar
c = load_cells(CLS)
assert len(e) == 28, f"Tema2_Etape23_Kaggle.ipynb are {len(e)} celule, nu 28 — regenereaza cu build_notebook_etape23.py inainte."

# ------------------------------------------------------------------ CONFIG
cfg = src(e[4])
cfg = must_replace(
    cfg, "CALC_MASK   = find_dir('Calcification_Only')",
    "CALC_MASK   = find_dir('Calcification_Only')\nMASK_DIR    = find_dir('Masks')")
cfg = must_replace(
    cfg, "FOLD      = 1",
    "FOLD      = 1\nPATCH_SIZE = 128    # latura patch-urilor pt. CLASIFICARE (Etapa 1)")
cfg = must_replace(
    cfg, "N_DET = 5  if SMOKE else 0    # 0 = toate imaginile",
    "N_DET = 5  if SMOKE else 0    # 0 = toate imaginile\n"
    "N_CLS = 500 if SMOKE else 0   # 0 = toate patch-urile (clasificare)")
cfg = must_replace(
    cfg, "('IMG_DIR', IMG_DIR), ('CALC_MASK', CALC_MASK),",
    "('IMG_DIR', IMG_DIR), ('MASK_DIR', MASK_DIR), ('CALC_MASK', CALC_MASK),")
cfg = must_replace(
    cfg, "SMOKE={SMOKE}  FOLD={FOLD}  TOP_FRAC={TOP_FRAC}  MIN_CROP={MIN_CROP}  ",
    "SMOKE={SMOKE}  FOLD={FOLD}  PATCH_SIZE={PATCH_SIZE}  TOP_FRAC={TOP_FRAC}  MIN_CROP={MIN_CROP}  ")

# ------------------------------------------------------- CLASIFICARE (Etapa 1)
cls_src = src(c[9])
# comutator SMOKE: subesantioneaza patch-urile
cls_src = must_replace(
    cls_src,
    "patches = [d['patch'] for d in data]; labels = [d['label'] for d in data]",
    "patches = [d['patch'] for d in data]; labels = [d['label'] for d in data]\n"
    "if N_CLS and len(patches) > N_CLS:\n"
    "    _r = np.random.RandomState(42)\n"
    "    _sel = _r.choice(len(patches), N_CLS, replace=False)\n"
    "    patches = [patches[i] for i in _sel]; labels = [labels[i] for i in _sel]\n"
    "    print(f'[cls] SMOKE: subesantionat la {len(patches)} patch-uri', flush=True)")
# progres in featurize (altfel pare blocat ~15 min)
cls_src = must_replace(
    cls_src, "        ch = pts[i:i+bs]",
    "        ch = pts[i:i+bs]\n"
    "        if (i // bs) % 10 == 0:\n"
    "            print(f'[cls] featurize {i}/{len(pts)}', flush=True)")
# feedback: pastreaza probabilitatile per-varianta (nu doar acuratetea), ca sa
# poata fi analizate per-clasa si per-caz mai jos.
cls_src = must_replace(
    cls_src,
    "res['cnn_trained'] = acc(train_head(tr['cnn_feat'], tr['labels'], va['cnn_feat'], yva, num_cls), yva)\n"
    "res['vlm_trained'] = acc(train_head(tr['vlm_emb'], tr['labels'], va['vlm_emb'], yva, num_cls), yva)\n"
    "res['feature_fusion'] = acc(train_head(ct, tr['labels'], cv, yva, num_cls), yva)",
    "p_cnn = train_head(tr['cnn_feat'], tr['labels'], va['cnn_feat'], yva, num_cls)\n"
    "p_vlm = train_head(tr['vlm_emb'], tr['labels'], va['vlm_emb'], yva, num_cls)\n"
    "p_fus = train_head(ct, tr['labels'], cv, yva, num_cls)\n"
    "res['cnn_trained'] = acc(p_cnn, yva)\n"
    "res['vlm_trained'] = acc(p_vlm, yva)\n"
    "res['feature_fusion'] = acc(p_fus, yva)")
# feedback #1/#2: de unde vine diferenta feature_fusion vs restul (per-clasa +
# per-caz, nu doar acuratetea agregata) + comparatie explicita cu celelalte
# variante CNN/VLM.
cls_src = must_replace(
    cls_src,
    "print('\\n===== REZULTATE CLASIFICARE =====')",
    "from sklearn.metrics import classification_report\n"
    "_CLS_NAMES = ['healthy', 'mass', 'calc', 'both'][:num_cls]\n"
    "per_class_f1 = {}\n"
    "for _nm, _p in [('cnn_trained', p_cnn), ('vlm_trained', p_vlm), ('feature_fusion', p_fus)]:\n"
    "    _rep = classification_report(yva, _p.argmax(1), target_names=_CLS_NAMES,\n"
    "                                 digits=4, zero_division=0, output_dict=True)\n"
    "    per_class_f1[_nm] = {c: round(_rep[c]['f1-score'], 4) for c in _CLS_NAMES}\n"
    "_ok_cnn, _ok_vlm, _ok_fus = p_cnn.argmax(1) == yva, p_vlm.argmax(1) == yva, p_fus.argmax(1) == yva\n"
    "case_breakdown = {\n"
    "    'fusion_salveaza_ambele_gresite': int((_ok_fus & ~_ok_cnn & ~_ok_vlm).sum()),\n"
    "    'fusion_salveaza_doar_cnn_gresit': int((_ok_fus & ~_ok_cnn & _ok_vlm).sum()),\n"
    "    'fusion_salveaza_doar_vlm_gresit': int((_ok_fus & _ok_cnn & ~_ok_vlm).sum()),\n"
    "    'fusion_strica_cnn_avea_dreptate': int((~_ok_fus & _ok_cnn).sum()),\n"
    "    'fusion_strica_vlm_avea_dreptate': int((~_ok_fus & _ok_vlm).sum()),\n"
    "    'fusion_gresit_ambele_gresite': int((~_ok_fus & ~_ok_cnn & ~_ok_vlm).sum()),\n"
    "    'n_total_val': int(len(yva)),\n"
    "}\n"
    "print('\\n===== De unde vine diferenta feature_fusion (F1 per clasa) =====')\n"
    "for _c in _CLS_NAMES:\n"
    "    print(f\"  {_c:10s}  cnn={per_class_f1['cnn_trained'][_c]:.3f}  \"\n"
    "          f\"vlm={per_class_f1['vlm_trained'][_c]:.3f}  fusion={per_class_f1['feature_fusion'][_c]:.3f}\")\n"
    "print('\\n===== Analiza pe cazuri (cate patch-uri castiga/pierde fuziunea) =====')\n"
    "print(json.dumps(case_breakdown, indent=2))\n"
    "\n"
    "print('\\n===== REZULTATE CLASIFICARE =====')")
# feedback #2: plot comparativ feature_fusion vs restul (cnn/vlm/late_fusion) +
# F1 per clasa + cazuri castigate/pierdute -> extinde summary.json cu noile campuri.
cls_src = must_replace(
    cls_src,
    "'agreement': ag}, open('/kaggle/working/VLM_Classification_Results/summary.json'",
    "'agreement': ag, 'per_class_f1': per_class_f1, 'case_breakdown': case_breakdown},\n"
    "         open('/kaggle/working/VLM_Classification_Results/summary.json'")
cls_src += '''

# ===== PLOT — clasificare: feature_fusion vs restul + F1 per clasa + cazuri =====
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 3, figsize=(18, 5))
_order = ['cnn_only_raw', 'vlm_zeroshot', 'late_fusion', 'cnn_trained', 'vlm_trained', 'feature_fusion']
_vals = [res[k] for k in _order]
_colors = ['#bbbbbb', '#bbbbbb', '#f4a261', '#4c72b0', '#e76f51', '#2a9d8f']
ax[0].bar(_order, _vals, color=_colors)
for i, v in enumerate(_vals):
    ax[0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)
ax[0].axhline(res['cnn_trained'], color='k', linestyle='--', linewidth=1, label='nivel cnn_trained')
ax[0].set_ylim(0, max(_vals) + 0.1)
ax[0].set_ylabel('acuratete')
ax[0].set_xticklabels(_order, rotation=30, ha='right')
ax[0].set_title('Feature fusion vs alte variante CNN/VLM')
ax[0].legend(fontsize=8)

_x = np.arange(len(_CLS_NAMES))
_w = 0.25
for i, (nm, col) in enumerate(zip(['cnn_trained', 'vlm_trained', 'feature_fusion'],
                                  ['#4c72b0', '#e76f51', '#2a9d8f'])):
    ax[1].bar(_x + (i - 1) * _w, [per_class_f1[nm][c] for c in _CLS_NAMES], _w, label=nm, color=col)
ax[1].set_xticks(_x); ax[1].set_xticklabels(_CLS_NAMES)
ax[1].set_ylabel('F1 per clasa')
ax[1].set_title('De unde vine diferenta: F1 per clasa')
ax[1].legend(fontsize=8)

ax[2].axis('off')
_txt = (f"Analiza pe cazuri (din {case_breakdown['n_total_val']} patch-uri val):\\n\\n" +
       '\\n'.join(f'{k} = {v}' for k, v in case_breakdown.items() if k != 'n_total_val'))
ax[2].text(0.02, 0.98, _txt, va='top', ha='left', family='monospace', fontsize=9)
ax[2].set_title('Cazuri castigate/pierdute de fuziune')

plt.tight_layout()
_p = '/kaggle/working/VLM_Classification_Results/classification_summary_plots.png'
plt.savefig(_p, dpi=130)
plt.close(fig)
print('[plot] salvat', _p, flush=True)'''

cls_schema_md = md("""## Schema — Feature Fusion Head (Etapa 1)

```
patch (128x128, grayscale)
        |
        +---------------------------+----------------------------+
        v                                                        v
 CNN (ResNet50, ImageNet)                          VLM (BiomedCLIP image encoder)
 resize->224, normalizare ImageNet                 preprocesare proprie BiomedCLIP
        |                                                        |
 features pre-logit (2048-d)                       embedding imagine normalizat (512-d)
        |                                                        |
        +------------------------------+-------------------------+
                                        v
                        concatenare [2048 || 512] = 2560-d
                                        |
                                        v
              MLP:  Linear(2560->512) -> BatchNorm -> ReLU -> Dropout(0.3)
                                        |
                                        v
                    Linear(512->4)  ->  softmax (healthy / mass / calc / both)
```

Ambele ramuri raman **inghetate** (nu se re-antreneaza ResNet50 sau BiomedCLIP);
se antreneaza DOAR capul MLP de mai sus, pe features/embedding-uri deja
cache-uite (`featurize()`, un singur pas peste toate patch-urile).

Acelasi cap (arhitectura identica, doar dimensiunea input-ului difera) se
antreneaza **separat** pe 3 seturi de intrare, ca sa izoleze contributia VLM:
- doar CNN (2048-d) -> `cnn_trained`
- doar VLM (512-d) -> `vlm_trained`
- concatenat (2560-d) -> `feature_fusion`

Pentru ca arhitectura capului, datele si seed-ul sunt identice intre cele 3
rulari, `feature_fusion - cnn_trained` izoleaza **exact** contributia
marginala a VLM (nu e confundata cu alta diferenta de setup).""")

cls_md = md("""## Etapa 1 — Clasificare (VLM + CNN pe patch-uri)

Compara pe patch-uri (4 clase: healthy / mass / calc / both), pe **feature-uri
inghetate** (comparatie corecta — acelasi cap MLP antrenat separat pe fiecare):
- `cnn_only_raw` (cap neantrenat, doar referinta), `vlm_zeroshot` (fara antrenare)
- `cnn_trained`, `vlm_trained`, `feature_fusion` (CNN+VLM concatenate)
- **contributia marginala VLM** = `feature_fusion - cnn_trained`

Rezultat asteptat (PATCH_SIZE=128, complet): fusion ~0.847 vs cnn ~0.825 -> **+2.2%**.
Cu `SMOKE=True` ruleaza pe `N_CLS` patch-uri (rapid, cifre orientative).

Dupa rulare, celula printeaza si salveaza (`classification_summary_plots.png`,
`summary.json`) **de unde vine** diferenta feature_fusion vs restul: F1 per
clasa (healthy/mass/calc/both) pentru cele 3 variante antrenate, plus o
analiza pe cazuri — cate patch-uri sunt salvate de fuziune (gresite de CNN
si/sau VLM separat, corecte in fuziune) vs. cate sunt stricate de fuziune.""")

# ------------------------------------------------------------- SUMAR FINAL
summary = code('''# ===== SUMAR FINAL (toate 3 etapele) =====
import json


def _load(p):
    try:
        return json.load(open(p))
    except FileNotFoundError:
        return None


print('===== ETAPA 1: CLASIFICARE =====')
cls = _load('/kaggle/working/VLM_Classification_Results/summary.json')
if cls:
    for k in ['cnn_only_raw', 'vlm_zeroshot', 'late_fusion',
              'cnn_trained', 'vlm_trained', 'feature_fusion']:
        if k in cls:
            print(f'  {k:16s} = {cls[k]:.4f}')
    if 'feature_fusion' in cls and 'cnn_trained' in cls:
        print(f"  >> contributie VLM (fusion - cnn) = "
              f"{cls['feature_fusion'] - cls['cnn_trained']:+.4f}")
    if 'per_class_f1' in cls:
        print('  F1 per clasa:', json.dumps(cls['per_class_f1'], indent=2))
    if 'case_breakdown' in cls:
        print('  cazuri castigate/pierdute de fuziune:', json.dumps(cls['case_breakdown'], indent=2))
else:
    print('  (lipseste — clasificarea nu a rulat)')

print('\\n===== ETAPA 2: DETECTIE =====')
d = _load(f'{OUT_DET}/detection_vlm_summary.json')
if d:
    print('baseline:', json.dumps(d['baseline'], indent=2))
    print('distributie scoruri VLM:',
          json.dumps(d['score_stats']['vlm_percentiles'], indent=2))
    print('diagnostic:', json.dumps(d.get('diagnostic', {}), indent=2))
    print('\\ngrila beta/min_vlm:')
    for g in d['grid']:
        print(f"  beta={g['beta']:<5} min_vlm={g['min_vlm']:<5} "
              f"kept={g['n_kept']:<6} {g['froc']}")
else:
    print('  (lipseste)')

print('\\n===== ETAPA 3: SEGMENTARE =====')
print(json.dumps(_load(f'{OUT_SEG}/segmentation_vlm_summary.json'), indent=2))

print('\\n===== GENERALIZARE (detectie pe mai multe fold-uri) =====')
gen = _load(f'{OUT_DET}/detection_multifold_summary.json')
if gen:
    print(json.dumps(gen.get('per_metric', {}), indent=2))
else:
    print('  (lipseste — celula de generalizare nu a rulat)')

print('\\nDocument de evaluare (cantitativ + calitativ, cu nume de imagini): '
      '/kaggle/working/EVALUARE_TEMA2.md')
print('Imagini de inspectat cu ochiul (masti/box-uri suprapuse):')
print('  /kaggle/working/qualitative/segmentare/  si  /kaggle/working/qualitative/detectie/')''')

# --------------------------------------------------------------- INTRO
intro = md("""# Tema 2 — VLM (BiomedCLIP) peste clasificare + detectie + segmentare

Notebook **COMPLET si self-contained**: toate cele 3 etape intr-o singura rulare,
de sus in jos. Driverele sunt inline; nu se aplica niciun patch pe fisiere.

**Setari panou dreapta:** Accelerator = **GPU**, Internet = **On**.

**Input-uri necesare (Add Input):**
- datasetul de mamografie (InBreast)
- datasetul cu cod (`breastcancer-vlm-code`) — trebuie sa contina si `folds.csv`
- datasetul cu checkpoint-uri (`retina_detection_best_model.pth`, `medsam_best.pth`)

Un singur comutator `SMOKE` in CONFIG: `True` = test rapid (cateva minute),
`False` = rulare completa. Ruleaza intai cu `True`.

Ordine: **Etapa 1 Clasificare → Etapa 2 Detectie (+ vizualizari + ablatie
prompturi + generalizare multi-fold) → Etapa 3 Segmentare (+ vizualizari) →
Sumar**. Celulele de vizualizare salveaza PNG-uri cu masti/box-uri suprapuse
in `/kaggle/working/qualitative/`, pentru inspectie vizuala directa (nu doar
cifre). Celula de generalizare repeta testul de detectie pe toate fold-urile
din `folds.csv`, ca sa se vada daca imbunatatirea VLM tine si in afara
fold-ului 1 sau e particulara acelui set.""")

# --------------------------------------------------------------- ASAMBLARE
cells = [
    intro,
    e[1], e[2], e[3],        # gpu, deps, copiaza cod
    code(cfg),               # CONFIG (extins)
    cls_schema_md, cls_md, code(cls_src),   # Etapa 1
    e[11], e[12],             # Etapa 2: det-md, det
    e[13], e[14],             # ploturi detectie (FROC, grila, distributie)
    e[15], e[16],             # viz calitativa detectie
    e[17], e[18],             # ablatie: md, cod
    e[19], e[20],             # plot ablatie
    e[21], e[22],             # generalizare multi-fold (detectie)
    e[23], e[24],             # plot generalizare
    e[5], e[6],                # Etapa 3: seg-md, seg
    e[7], e[8],                # viz calitativa segmentare
    e[9], e[10],                # ploturi agregate segmentare
    e[25], e[26],               # document de evaluare (cantitativ+calitativ, cu nume de imagini)
    summary,
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}

with io.open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=True)
print("scris:", OUT, os.path.getsize(OUT), "bytes")

# validare: json valid + fiecare celula de cod compileaza
nb2 = json.load(io.open(OUT, encoding="utf-8"))
n_ok = 0
for i, cell in enumerate(nb2["cells"]):
    if cell["cell_type"] == "code":
        s = src(cell)
        s = "\n".join(l for l in s.split("\n") if not l.strip().startswith("!"))
        try:
            ast.parse(s)
        except SyntaxError as ex:
            raise SyntaxError(f"Celula {i} nu compileaza: {ex}")
        n_ok += 1
print("celule de cod validate sintactic:", n_ok, "/ total celule:", len(nb2["cells"]))
