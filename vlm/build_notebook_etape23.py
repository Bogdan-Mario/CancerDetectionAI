import json, io, os

def md(src):  return {"cell_type": "markdown", "metadata": {}, "source": src}
def code(src): return {"cell_type": "code", "execution_count": None, "metadata": {},
                       "outputs": [], "source": src}

cells = []

cells.append(md("""# Tema 2 — Etapa 2 (Detectie) + Etapa 3 (Segmentare)

Notebook **self-contained**: driverele sunt scrise inline, nu se aplica niciun patch
pe fisiere. Ruleaza celulele in ordine, de sus in jos.

**Setari panou dreapta:** Accelerator = **GPU**, Internet = **On**.

**Input-uri necesare (Add Input):**
- datasetul de mamografie (InBreast)
- datasetul cu cod (`breastcancer-vlm-code`) — trebuie sa contina si `folds.csv`
- datasetul cu checkpoint-uri (`retina_detection_best_model.pth`, `medsam_best.pth`)

Setarea `SMOKE` din celula CONFIG controleaza tot: `True` = test rapid (~5 min),
`False` = rulare completa (~1.5 h). Ruleaza intai cu `True`."""))

cells.append(code("""# 0. GPU + ce input-uri sunt atasate
import torch, os
print('CUDA:', torch.cuda.is_available(), '-',
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'FARA GPU')
print('Input-uri:')
for d in sorted(os.listdir('/kaggle/input')):
    print('  ', d)"""))

cells.append(code("""# 1. Dependinte (~3 min)
!pip -q install open_clip_torch transformers timm efficientnet_pytorch torchmetrics albumentations
!pip -q install 'git+https://github.com/facebookresearch/segment-anything.git'
print('gata dependinte')"""))

cells.append(code("""# 2. Copiaza codul Tema 1 + pachetul vlm/ in /kaggle/working/code
import os, shutil, sys
code_src = None
for root, dirs, files in os.walk('/kaggle/input'):
    if 'vlm' in dirs and 'PatchFilterClassificationTrainingAndTesting.py' in files:
        code_src = root
        break
assert code_src, 'Nu am gasit codul. Ai atasat datasetul cu cod?'
CODE_DIR = '/kaggle/working/code'
if os.path.exists(CODE_DIR):
    shutil.rmtree(CODE_DIR)
shutil.copytree(code_src, CODE_DIR)
os.chdir(CODE_DIR)
sys.path.insert(0, CODE_DIR)
print('cod:', code_src, '->', CODE_DIR)"""))

cells.append(code('''# 3. CONFIG — cai auto-descoperite + parametri
import os, shutil

def find(fname, max_depth=6):
    """Cauta un fisier dupa nume, in latime, oprindu-se la max_depth (rapid pe FUSE)."""
    q = [('/kaggle/input', 0)]
    while q:
        d, depth = q.pop(0)
        try:
            entries = list(os.scandir(d))
        except OSError:
            continue
        for e in entries:
            if e.is_file() and e.name == fname:
                return e.path
        if depth < max_depth:
            q += [(e.path, depth + 1) for e in entries if e.is_dir()]
    return ''

def find_dir(dname, max_depth=6):
    q = [('/kaggle/input', 0)]
    while q:
        d, depth = q.pop(0)
        try:
            entries = [e for e in os.scandir(d) if e.is_dir()]
        except OSError:
            continue
        for e in entries:
            if e.name == dname:
                return e.path
        if depth < max_depth:
            q += [(e.path, depth + 1) for e in entries]
    return ''

IMG_DIR     = find_dir('Normal')
CALC_MASK   = find_dir('Calcification_Only')
RETINA_CKPT = find('retina_detection_best_model.pth')
MEDSAM_CKPT = find('medsam_best.pth')

FOLDS_CSV = os.path.join(CODE_DIR, 'folds.csv')
if not os.path.exists(FOLDS_CSV):
    _src = find('folds.csv')
    if _src:
        shutil.copy(_src, FOLDS_CSV)
        print('folds.csv copiat din', _src)

# ---------------- parametri ----------------
SMOKE     = True    # True = test rapid; False = rulare completa
FOLD      = 1
TOP_FRAC  = 0.02    # fractia de pixeli cei mai relevanti pastrata din harta VLM
WIN, STRIDE = 32, 16
MIN_CROP  = 128     # latura minima a crop-ului trimis la VLM (context anatomic).
                    # Boxurile de microcalcificari au latura mediana ~16 px; taiate
                    # strans si marite la 224 devin pete fara continut semantic.
SEG_CTX   = 96      # la segmentare: fereastra fina (WIN) da localizarea, dar
                    # embedding-ul VLM se face pe un crop de SEG_CTX px centrat pe
                    # fereastra -> acelasi fix de context care a reparat detectia
                    # (AUC 0.326 -> 0.702). Pe patch de 256, 96 pastreaza si context
                    # si variatie de pozitie (crop-ul de 128/MIN_CROP = tot patch-ul).
N_SEG = 40 if SMOKE else 0    # 0 = toate patch-urile
N_DET = 5  if SMOKE else 0    # 0 = toate imaginile
OUT_SEG = '/kaggle/working/VLM_Segmentation_Results'
OUT_DET = '/kaggle/working/VLM_Detection_Results'
os.makedirs(OUT_SEG, exist_ok=True)
os.makedirs(OUT_DET, exist_ok=True)

_missing = []
for _n, _p in [('IMG_DIR', IMG_DIR), ('CALC_MASK', CALC_MASK), ('FOLDS_CSV', FOLDS_CSV),
               ('RETINA_CKPT', RETINA_CKPT), ('MEDSAM_CKPT', MEDSAM_CKPT)]:
    ok = bool(_p) and os.path.exists(_p)
    print(('OK   ' if ok else 'LIPSA'), f'{_n:12s} ->', _p if _p else '(negasit)')
    if not ok:
        _missing.append(_n)
assert not _missing, f'Lipsesc: {_missing}. Verifica input-urile atasate.'
print(f'\\nSMOKE={SMOKE}  FOLD={FOLD}  TOP_FRAC={TOP_FRAC}  MIN_CROP={MIN_CROP}  '
      f'N_SEG={N_SEG}  N_DET={N_DET}')'''))

cells.append(md("""## Etapa 3 — Segmentare (MedSAM ghidat de VLM)

Compara **trei** surse de prompt (box) pentru MedSAM, toate pe aceleasi patch-uri:
1. **`dice_gt_box`** — box-ul GT. Plafonul superior; arata cat de bun e MedSAM in sine.
2. **`dice_vlm_box`** — box din harta de relevanta a VLM-ului, ACUM cu **context**:
   fereastra fina (`WIN`) da localizarea, dar embedding-ul se face pe un crop de
   `SEG_CTX` px centrat pe fereastra + prompturi `fara_artifact`. Exact fix-ul care
   a inversat semnalul la detectie (AUC 0.326 -> 0.702). Pragul e o **percentila**
   (`TOP_FRAC`), box-ul din **componenta conexa cea mai mare**.
3. **`dice_det_box`** — box-ul de la RetinaNet **rerankuit de VLM** (detectia reparata),
   proiectat in patch. Leaga Etapa 2 de Etapa 3: detectorul care ACUM merge devine
   sursa de prompt. De obicei cea mai buna Dice practica.

Rulare anterioara (fara context): `dice_vlm_box`=0.255, IoU relevanta=0.028 — VLM nu
localiza. Aici testam daca contextul + prompturile bune repara si segmentarea.

**Ce foloseste fiecare scor (sa fie clar; nu exista "VLM antrenat" la segmentare —
BiomedCLIP ramane zero-shot peste tot, spre deosebire de Etapa 1 unde exista si
un cap antrenat):**

| metrica | sursa box-ului | VLM zero-shot? | ce arata |
|---|---|---|---|
| `dice_gt_box` | box GT (adnotare umana) | nu foloseste VLM | plafonul MedSAM |
| `dice_vlm_box` | harta de relevanta BiomedCLIP (prompturi `fara_artifact`, fereastra `WIN` + context `SEG_CTX`) | da, zero-shot | cat de bine localizeaza VLM-ul SINGUR, fara detector |
| `dice_det_box` | box RetinaNet, rerankuit dupa scor VLM (acelasi zero-shot, dar aplicat pe boxurile detectorului, nu pe fereastra glisanta) | da, zero-shot | cat de bine ajuta VLM-ul cand porneste de la propunerile detectorului |"""))

cells.append(code('''# ===== ETAPA 3: SEGMENTARE (inline) =====
import os, sys, json, time, numpy as np, cv2, torch, torch.nn.functional as Fnn
sys.path.insert(0, CODE_DIR)
import MedSam2 as seg
from segment_anything import sam_model_registry
from vlm.vlm_core import VLM
from vlm.clinical_prompts import DETECTION_POS_PROMPTS

seg.SOURCE_IMG_DIR, seg.SOURCE_MASK_DIR, seg.FOLDS_CSV = IMG_DIR, CALC_MASK, FOLDS_CSV
seg.seed_all(seg.SEED)

_, _, test_ids = seg.load_fold(FOLD)
print(f'[seg] {len(test_ids)} imagini in test; construiesc patch-urile...', flush=True)
samples = seg.build_samples(test_ids)
if N_SEG:
    samples = samples[:N_SEG]
print(f'[seg] {len(samples)} patch-uri lezionale', flush=True)

sam = sam_model_registry['vit_b'](checkpoint=None)
model = seg.MedSAM(sam).to(seg.DEVICE)
model.load_state_dict(torch.load(MEDSAM_CKPT, map_location=seg.DEVICE))
model.eval()
print('[seg] MedSAM incarcat', flush=True)
vlm = VLM(device=seg.DEVICE)

# Bank pos/neg cu negativele "fara_artifact" (au fost cele mai informative la
# detectie, AUC 0.702). Le refolosim ca semnal semantic si la segmentare.
SEG_POS_PROMPTS = DETECTION_POS_PROMPTS
SEG_NEG_PROMPTS = ['normal breast tissue without calcifications',
                   'empty background outside the breast']
_pos = Fnn.normalize(vlm.encode_text(SEG_POS_PROMPTS).mean(0, keepdim=True), dim=-1)
_neg = Fnn.normalize(vlm.encode_text(SEG_NEG_PROMPTS).mean(0, keepdim=True), dim=-1)
SEG_BANK = Fnn.normalize(torch.cat([_pos, _neg], 0), dim=-1)   # (2, D)


def run_medsam(patch, box_patch):
    """Un patch + un box in coordonate de patch -> masca binara (P,P)."""
    P, S = seg.PATCH_SIZE, seg.SAM_SIZE
    img1024 = cv2.resize(patch, (S, S))
    x = torch.from_numpy(np.stack([img1024] * 3, 0).astype(np.float32) / 255.0)[None]
    box1024 = torch.as_tensor(np.asarray(box_patch, np.float32) * (S / P))[None]
    with torch.no_grad():
        logits = model(x.to(seg.DEVICE), box1024.to(seg.DEVICE))
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    return (cv2.resize(prob, (P, P)) > seg.THRESHOLD).astype(np.uint8)


P = seg.PATCH_SIZE


@torch.no_grad()
def vlm_box(patch):
    """Harta de relevanta CU CONTEXT -> (box, masca_prag, heat).

    Fereastra fina WIN da localizarea; embedding-ul se face pe un crop de SEG_CTX
    px centrat pe fereastra (context anatomic, ca la detectie). Prompturi
    pos-vs-neg 'fara_artifact'. Box din cea mai mare componenta conexa."""
    H, W = patch.shape[:2]
    heat = np.zeros((H, W), np.float32)
    cnt = np.zeros((H, W), np.float32)
    coords, crops = [], []
    half = SEG_CTX // 2
    for y in range(0, max(1, H - WIN + 1), STRIDE):
        for x in range(0, max(1, W - WIN + 1), STRIDE):
            cx, cy = x + WIN // 2, y + WIN // 2
            X1, X2 = max(0, cx - half), min(W, cx + half)
            Y1, Y2 = max(0, cy - half), min(H, cy + half)
            crops.append(patch[Y1:Y2, X1:X2])
            coords.append((x, y))
    if not crops:
        return None, np.zeros((H, W), np.uint8), heat
    e = vlm.encode_image(crops)
    s = vlm.similarity(e, SEG_BANK, softmax=True)[:, 0].cpu().numpy()   # P(pozitiv)
    for (x, y), sc in zip(coords, s):
        heat[y:y + WIN, x:x + WIN] += sc
        cnt[y:y + WIN, x:x + WIN] += 1.0
    cnt[cnt == 0] = 1.0
    heat /= cnt
    mn, mx = heat.min(), heat.max()
    heat = (heat - mn) / (mx - mn + 1e-6)
    q = float(np.quantile(heat, 1.0 - TOP_FRAC))
    m = (heat >= q).astype(np.uint8)
    n, lab = cv2.connectedComponents(m)
    if n <= 1:
        return None, m, heat
    sizes = [int((lab == i).sum()) for i in range(1, n)]
    big = 1 + int(np.argmax(sizes))
    ys, xs = np.where(lab == big)
    return ([int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)],
            (lab == big).astype(np.uint8), heat)


# ---- Fix (b): box de la RetinaNet rerankuit de VLM, folosit ca prompt MedSAM ----
import detectionRetinaNet as det
det.SOURCE_IMG_DIR, det.SOURCE_MASK_DIR, det.FOLDS_CSV = IMG_DIR, CALC_MASK, FOLDS_CSV
det.ENCODER_INIT = 'imagenet'
det_model = det.build_model()
det_model.load_state_dict(torch.load(RETINA_CKPT, map_location=det.DEVICE))
det_model.eval()
print('[seg] RetinaNet incarcat (prompturi det->seg)', flush=True)


def _crop_ctx(img, box, min_side):
    """Crop patrat centrat pe box, latura >= min_side (context pt. VLM)."""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = [float(t) for t in box]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half = max(float(min_side), x2 - x1, y2 - y1) / 2.0
    X1, X2 = int(max(0, round(cx - half))), int(min(W, round(cx + half)))
    Y1, Y2 = int(max(0, round(cy - half))), int(min(H, round(cy + half)))
    if X2 <= X1 or Y2 <= Y1:
        return None
    return img[Y1:Y2, X1:X2]


@torch.no_grad()
def _det_boxes_scored(iid):
    """Boxuri RetinaNet (coord imagine) + scor VLM per box (crop cu context)."""
    img = seg._img(iid)
    boxes, _ = det.predict_image(det_model, iid)
    b = (boxes.cpu().numpy() if torch.is_tensor(boxes) else np.asarray(boxes)).reshape(-1, 4)
    v = np.zeros(len(b), np.float32)
    if len(b) == 0:
        return b, v
    crops = [_crop_ctx(img, x, MIN_CROP) for x in b]
    keep = [i for i, c in enumerate(crops) if c is not None and min(c.shape[:2]) >= 4]
    if keep:
        e = vlm.encode_image([crops[i] for i in keep])
        p = vlm.similarity(e, SEG_BANK, softmax=True)[:, 0].cpu().numpy()
        v[np.asarray(keep)] = p
    return b, v


_det_cache = {}


def det_prompt_for_patch(iid, px, py):
    """Cel mai bun box de detectie (dupa scor VLM) al carui centru cade in patch,
    proiectat in coordonate de patch. None daca niciunul."""
    if iid not in _det_cache:
        _det_cache[iid] = _det_boxes_scored(iid)
    b, v = _det_cache[iid]
    if len(b) == 0:
        return None
    cx, cy = (b[:, 0] + b[:, 2]) / 2.0, (b[:, 1] + b[:, 3]) / 2.0
    inside = (cx >= px) & (cx < px + P) & (cy >= py) & (cy < py + P)
    if not inside.any():
        return None
    idx = np.where(inside)[0]
    best = idx[int(np.argmax(v[idx]))]
    bx = b[best].astype(np.float32).copy()
    bx[[0, 2]] -= px
    bx[[1, 3]] -= py
    bx[:2] = np.clip(bx[:2], 0, P - 1)
    bx[2:] = np.clip(bx[2:], 1, P)
    if bx[2] <= bx[0] or bx[3] <= bx[1]:
        return None
    return [float(t) for t in bx]


def dice(a, b):
    inter = np.logical_and(a > 0, b > 0).sum()
    s = (a > 0).sum() + (b > 0).sum()
    return float(2 * inter / s) if s > 0 else 1.0


d_gt, d_vlm, d_det, ious, areas_v, areas_g = [], [], [], [], [], []
has_det_box = []   # feedback: distinge "fara box de detectie" de "box prost localizat"
n_found = n_det = 0
t0 = time.time()
for i, (iid, px, py, gt_box) in enumerate(samples):
    if i % 25 == 0:
        print(f'[seg] {i}/{len(samples)}  {time.time()-t0:.0f}s', flush=True)
    patch = seg.pad(seg._img(iid)[py:py + P, px:px + P], P)
    gmask = seg.pad((seg._mask(iid)[py:py + P, px:px + P] > 127).astype(np.uint8), P)

    m_gt = run_medsam(patch, gt_box)
    d_gt.append(dice(m_gt, gmask))
    areas_g.append(((gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])) / float(P * P))

    # (2) box din harta de relevanta cu context
    box, region, heat = vlm_box(patch)
    if box is None:
        d_vlm.append(0.0)
    else:
        n_found += 1
        m_vlm = run_medsam(patch, box)
        d_vlm.append(dice(m_vlm, gmask))
        inter = np.logical_and(m_vlm > 0, region > 0).sum()
        union = np.logical_or(m_vlm > 0, region > 0).sum()
        ious.append(float(inter / union) if union > 0 else 0.0)
        areas_v.append(((box[2] - box[0]) * (box[3] - box[1])) / float(P * P))

    # (3) box de la detectie rerankuita
    db = det_prompt_for_patch(iid, px, py)
    if db is None:
        d_det.append(0.0)
        has_det_box.append(False)
    else:
        n_det += 1
        d_det.append(dice(run_medsam(patch, db), gmask))
        has_det_box.append(True)

mean = lambda a: float(np.mean(a)) if len(a) else 0.0
seg_summary = {
    'smoke': SMOKE, 'fold': FOLD, 'top_frac': TOP_FRAC, 'win': WIN, 'stride': STRIDE,
    'seg_ctx': SEG_CTX, 'min_crop': MIN_CROP,
    'n_patches': len(samples), 'n_vlm_prompted': n_found, 'n_det_prompted': n_det,
    'dice_gt_box': mean(d_gt), 'dice_vlm_box': mean(d_vlm), 'dice_det_box': mean(d_det),
    'mean_mask_relevance_iou': mean(ious),
    'mean_vlm_box_area_frac': mean(areas_v), 'mean_gt_box_area_frac': mean(areas_g),
    'minutes': round((time.time() - t0) / 60, 1),
}
with open(os.path.join(OUT_SEG, 'segmentation_vlm_summary.json'), 'w') as f:
    json.dump(seg_summary, f, indent=2, default=float)
print(json.dumps(seg_summary, indent=2))'''))

cells.append(md("""## Vizualizari calitative — segmentare

Cifra de Dice singura nu arata DE CE un box e slab. Salvez imagini de inspectie:
patch original, masca GT, harta de relevanta VLM cu box-ul extras, masca MedSAM
din box-ul VLM, masca MedSAM din box-ul de detectie rerankuit.

**Selectia nu mai e uniforma** (feedback: exemplele trebuie ALESE, nu doar
afisate la intamplare, ca sa arate diferenta): calculez Dice pe toate
patch-urile INTAI, apoi aleg explicit cele mai bune si cele mai proaste cazuri
dupa `dice_det_box` (sursa practica castigatoare) — plus cateva la mijloc
pentru context. Fiecare PNG e etichetat `bun`/`prost`/`mediu`/`fara_box_det`
si arata si prompturile text folosite (concluzia trebuie sustinuta de
exemple, nu doar de cifra agregata).

**Corectie**: categoria `prost` se alege DOAR dintre patch-urile unde
RetinaNet a propus un box (`has_det_box`) — nu si dintre cele fara niciun box
propus (acelea au `dice_det_box=0.0` prin constructie si ar fi dominat gresit
coada "proaste" fara sa arate o localizare rea). Patch-urile fara box propus
apar separat, cu categoria `fara_box_det`."""))

cells.append(code('''# ===== VIZUALIZARI: SEGMENTARE (exemple BUNE/PROASTE, nu esantion uniform) =====
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

VIZ_SEG_DIR = '/kaggle/working/qualitative/segmentare'
os.makedirs(VIZ_SEG_DIR, exist_ok=True)

# Dice pe TOATE patch-urile e deja calculat mai sus (d_gt/d_vlm/d_det, aliniat
# cu `samples`) -> selectez dupa el, nu re-calculez.
# IMPORTANT: "prost" trebuie sa insemne box GASIT dar prost localizat, nu
# "RetinaNet n-a propus niciun box in patch" (acelea au dice_det_box=0.0 prin
# constructie si ar domina complet coada "prost" fara sa arate o localizare
# gresita). Ranking-ul bun/prost se face DOAR pe patch-urile cu box gasit
# (`has_det_box`); cele fara box intra separat, la "mediu", ca sa fie totusi
# vizibile, dar nu ca dovada de localizare rea.
_have = [i for i in range(len(samples)) if has_det_box[i]]
_no_box = [i for i in range(len(samples)) if not has_det_box[i]]
_order_by_det = np.array(sorted(_have, key=lambda i: d_det[i]))   # crescator: cele mai proaste primele
N_VIZ_SEG = min(18, len(samples))
_n_bad = min(7, len(_order_by_det))
_n_good = min(7, len(_order_by_det))
_bad_idx = list(_order_by_det[:_n_bad])                 # dice_det_box cel mai mic (box GASIT, prost localizat)
_good_idx = [i for i in _order_by_det[::-1] if i not in _bad_idx][:_n_good]  # cel mai mare, fara suprapunere
_rest = N_VIZ_SEG - len(set(_bad_idx) | set(_good_idx))
_mid_pool = (list(np.random.RandomState(42).permutation(_no_box)) +
            [i for i in _order_by_det if i not in _bad_idx and i not in _good_idx])
_mid_idx = list(np.linspace(0, len(_mid_pool) - 1, max(_rest, 0)).astype(int)) if _mid_pool and _rest > 0 else []
_mid_idx = [_mid_pool[j] for j in _mid_idx]
_no_box_set = set(_no_box)
_viz_idx, _viz_cat = [], []
for i in _bad_idx:
    _viz_idx.append(i); _viz_cat.append('prost')
for i in _mid_idx:
    _viz_idx.append(i); _viz_cat.append('fara_box_det' if i in _no_box_set else 'mediu')
for i in _good_idx:
    _viz_idx.append(i); _viz_cat.append('bun')

_prompt_caption = ('prompturi pozitive: ' + '; '.join(SEG_POS_PROMPTS) +
                   '  |  prompturi negative: ' + '; '.join(SEG_NEG_PROMPTS))

# seg_viz_records tine, per imagine PNG salvata, numele imaginii (iid) + patch-ul +
# cele 3 Dice-uri + categoria (bun/mediu/prost) -> se foloseste in documentul de
# evaluare, ca sa se poata compara ulterior aceleasi imagini intre metode.
seg_viz_records = []
for k, (i, cat) in enumerate(zip(_viz_idx, _viz_cat)):
    iid, px, py, gt_box = samples[i]
    patch = seg.pad(seg._img(iid)[py:py + P, px:px + P], P)
    gmask = seg.pad((seg._mask(iid)[py:py + P, px:px + P] > 127).astype(np.uint8), P)
    m_gt = run_medsam(patch, gt_box)
    box, region, heat = vlm_box(patch)
    m_vlm = run_medsam(patch, box) if box is not None else np.zeros_like(gmask)
    db = det_prompt_for_patch(iid, px, py)
    m_det = run_medsam(patch, db) if db is not None else np.zeros_like(gmask)
    dg, dv, dd = dice(m_gt, gmask), dice(m_vlm, gmask), dice(m_det, gmask)

    fig, ax = plt.subplots(1, 5, figsize=(18, 4.3))
    ax[0].imshow(patch, cmap='gray'); ax[0].set_title(f'{iid} patch'); ax[0].axis('off')
    ax[1].imshow(patch, cmap='gray'); ax[1].imshow(gmask, cmap='Reds', alpha=0.4)
    ax[1].set_title(f'GT mask (dice_gt={dg:.2f})'); ax[1].axis('off')
    ax[2].imshow(heat, cmap='jet')
    if box is not None:
        x1, y1, x2, y2 = box
        ax[2].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='white', linewidth=2))
    ax[2].set_title('harta relevanta VLM + box'); ax[2].axis('off')
    ax[3].imshow(patch, cmap='gray'); ax[3].imshow(m_vlm, cmap='Blues', alpha=0.4)
    ax[3].set_title(f'MedSAM(box VLM) dice={dv:.2f}'); ax[3].axis('off')
    ax[4].imshow(patch, cmap='gray'); ax[4].imshow(m_det, cmap='Greens', alpha=0.4)
    ax[4].set_title(f'MedSAM(box detectie) dice={dd:.2f}'); ax[4].axis('off')
    fig.suptitle(f'[{cat.upper()}]  {_prompt_caption}', fontsize=8)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fname = f'{k:02d}_{cat}_{iid}_{px}_{py}.png'
    plt.savefig(os.path.join(VIZ_SEG_DIR, fname), dpi=110)
    plt.close(fig)
    seg_viz_records.append({'idx': k, 'category': cat, 'image_id': str(iid), 'patch_xy': [int(px), int(py)],
                            'file': fname, 'dice_gt_box': round(float(dg), 4),
                            'dice_vlm_box': round(float(dv), 4), 'dice_det_box': round(float(dd), 4),
                            'prompts_pos': list(SEG_POS_PROMPTS), 'prompts_neg': list(SEG_NEG_PROMPTS)})

with open(os.path.join(VIZ_SEG_DIR, 'index.json'), 'w') as f:
    json.dump(seg_viz_records, f, indent=2)
_n_no_box_shown = sum(1 for c in _viz_cat if c == 'fara_box_det')
print(f'[viz] {len(_viz_idx)} imagini salvate in {VIZ_SEG_DIR} '
      f'({_n_bad} proaste, {len(_mid_idx) - _n_no_box_shown} medii, '
      f'{_n_no_box_shown} fara_box_det, {_n_good} bune)', flush=True)'''))

cells.append(md("""## Ploturi agregate — segmentare

Exemplele calitative de mai sus arata cate un patch. Aici agreg **toate**
patch-urile din rulare (`n_patches`, nu doar cele `N_VIZ_SEG` afisate): dice
mediu +/- std pe cele 3 surse de prompt, distributia dice per patch (histograma,
nu un singur numar), si aria box-ului VLM vs aria box-ului GT."""))

cells.append(code('''# ===== PLOTURI AGREGATE — segmentare (pe toate patch-urile) =====
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 2, figsize=(13, 10))

# 1) bar mean +/- std pentru cele 3 surse de prompt
names = ['gt_box\\n(plafon)', 'vlm_box\\n(relevanta)', 'det_box\\n(detectie+VLM)']
vals = [d_gt, d_vlm, d_det]
means = [float(np.mean(v)) if len(v) else 0.0 for v in vals]
stds = [float(np.std(v)) if len(v) else 0.0 for v in vals]
colors = ['#888888', '#4c72b0', '#55a868']
ax[0, 0].bar(names, means, yerr=stds, capsize=6, color=colors)
for i, m in enumerate(means):
    ax[0, 0].text(i, m + 0.02, f'{m:.3f}', ha='center', fontweight='bold')
ax[0, 0].set_ylim(0, 1.05)
ax[0, 0].set_ylabel('Dice')
ax[0, 0].set_title(f'Dice mediu ± std pe {len(samples)} patch-uri')

# 2) histograme suprapuse ale Dice per patch
bins = np.linspace(0, 1, 21)
ax[0, 1].hist(d_gt, bins=bins, alpha=0.5, label='gt_box', color=colors[0])
ax[0, 1].hist(d_vlm, bins=bins, alpha=0.5, label='vlm_box', color=colors[1])
ax[0, 1].hist(d_det, bins=bins, alpha=0.5, label='det_box', color=colors[2])
ax[0, 1].set_xlabel('Dice per patch'); ax[0, 1].set_ylabel('numar patch-uri')
ax[0, 1].set_title('Distributia Dice pe toate patch-urile')
ax[0, 1].legend()

# 3) scatter: aria box VLM vs aria box GT (fractiune din patch)
n = min(len(areas_v), len(areas_g))
if n:
    ax[1, 0].scatter(areas_g[:n], areas_v[:n], s=14, alpha=0.5, color=colors[1])
    lim = max(max(areas_g[:n], default=0.01), max(areas_v[:n], default=0.01), 0.01) * 1.1
    ax[1, 0].plot([0, lim], [0, lim], 'k--', linewidth=1, label='aria egala')
    ax[1, 0].set_xlabel('aria box GT / aria patch')
    ax[1, 0].set_ylabel('aria box VLM / aria patch')
    ax[1, 0].set_title('Aria box VLM vs aria box GT')
    ax[1, 0].legend()
else:
    ax[1, 0].axis('off')

# 4) rezumat text
ax[1, 1].axis('off')
txt = (f"n_patches = {seg_summary['n_patches']}\\n"
       f"n_vlm_prompted = {seg_summary['n_vlm_prompted']}\\n"
       f"n_det_prompted = {seg_summary['n_det_prompted']}\\n\\n"
       f"dice_gt_box  = {seg_summary['dice_gt_box']:.3f}\\n"
       f"dice_vlm_box = {seg_summary['dice_vlm_box']:.3f}\\n"
       f"dice_det_box = {seg_summary['dice_det_box']:.3f}\\n\\n"
       f"mean_mask_relevance_iou = {seg_summary['mean_mask_relevance_iou']:.4f}\\n"
       f"mean_vlm_box_area_frac  = {seg_summary['mean_vlm_box_area_frac']:.4f}\\n"
       f"mean_gt_box_area_frac   = {seg_summary['mean_gt_box_area_frac']:.4f}")
ax[1, 1].text(0.02, 0.98, txt, va='top', ha='left', family='monospace', fontsize=11)

plt.tight_layout()
_p = os.path.join(OUT_SEG, 'segmentation_summary_plots.png')
plt.savefig(_p, dpi=130)
plt.close(fig)
print('[plot] salvat', _p, flush=True)'''))

cells.append(md("""## Etapa 2 — Detectie (RetinaNet + validare/reranking VLM)

Scorarea VLM a fiecarui box se face **o singura data**; fuziunea `beta`/`min_vlm`
e ieftina, deci se calculeaza FROC pentru o **grila intreaga** de setari in loc de
una singura ghicita. Scorurile brute se salveaza in `raw_detections.npz`, ca sa se
poata re-analiza fara GPU.

Recuperarea de fals-negative (`recover_missed_boxes`) e **exclusa** intentionat:
foloseste aceeasi harta de relevanta saturata, cu prag absolut, si ar propune sute
de box-uri pe imagine. Se adauga dupa ce grila arata punctul bun de operare.

Pe langa FROC (sens@Xfp), raportez si **AP@IoU=0.5** (`ap50`, metrica standard
mAP la un singur prag) si **Dice mediu** (`dice_boxes`, suprapunerea reala
box-GT, nu doar IoU>=prag) — pentru baseline si pentru fiecare punct din grila
beta/min_vlm."""))

cells.append(code('''# ===== ETAPA 2: DETECTIE (inline) =====
import os, sys, json, time, numpy as np, torch, torch.nn.functional as Fnn
sys.path.insert(0, CODE_DIR)
import detectionRetinaNet as det
from vlm.fusion_ops import crop_box, rerank_and_filter
from vlm.clinical_prompts import DETECTION_POS_PROMPTS, DETECTION_NEG_PROMPTS

det.SOURCE_IMG_DIR, det.SOURCE_MASK_DIR, det.FOLDS_CSV = IMG_DIR, CALC_MASK, FOLDS_CSV
det.ENCODER_INIT = 'imagenet'   # arhitectura e identica; greutatile vin din CKPT
det.seed_all(det.SEED)

_, _, test_ids = det.load_fold(FOLD)
if N_DET:
    test_ids = test_ids[:N_DET]
print(f'[det] {len(test_ids)} imagini de procesat', flush=True)

model = det.build_model()
model.load_state_dict(torch.load(RETINA_CKPT, map_location=det.DEVICE))
model.eval()
print('[det] RetinaNet incarcat', flush=True)
try:
    vlm
except NameError:
    from vlm.vlm_core import VLM
    vlm = VLM(device=det.DEVICE)


def crop_ctx(img, box, min_side=None):
    """Crop patrat centrat pe box, cu latura >= min_side, ca VLM-ul sa vada context."""
    if min_side is None:
        min_side = MIN_CROP
    H, W = img.shape[:2]
    x1, y1, x2, y2 = [float(t) for t in box]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half = max(float(min_side), x2 - x1, y2 - y1) / 2.0
    X1, X2 = int(max(0, round(cx - half))), int(min(W, round(cx + half)))
    Y1, Y2 = int(max(0, round(cy - half))), int(min(H, round(cy + half)))
    if X2 <= X1 or Y2 <= Y1:
        return None
    return img[Y1:Y2, X1:X2]


# Ablatia pe rularea full (82 img, fold 1) a aratat ca promptul negativ original
# ("an imaging artifact, not a calcification") descria involuntar pozitivul:
# o microcalcificare stralucitoare = un clip metalic. Setul 'fara_artifact' a urcat
# AUC de la 0.543 la 0.702. Il folosim ca negativ pentru FROC.
DET_NEG_PROMPTS = ['normal breast tissue without calcifications',
                   'empty background outside the breast']
POS_BANK = Fnn.normalize(vlm.encode_text(DETECTION_POS_PROMPTS).mean(0, keepdim=True), dim=-1)
NEG_BANK = Fnn.normalize(vlm.encode_text(DET_NEG_PROMPTS).mean(0, keepdim=True), dim=-1)
BANK = Fnn.normalize(torch.cat([POS_BANK, NEG_BANK], 0), dim=-1)
EMB_DIM = POS_BANK.shape[1]


@torch.no_grad()
def vlm_scores_for_boxes(img, boxes):
    """P(microcalcificare reala) per box + embedding-urile crop-urilor (pt. ablatii)."""
    boxes = np.asarray(boxes).reshape(-1, 4)
    out = np.zeros((len(boxes),), np.float32)
    embs = np.zeros((len(boxes), EMB_DIM), np.float32)
    if len(boxes) == 0:
        return out, embs
    crops = [crop_ctx(img, b) if MIN_CROP else crop_box(img, b) for b in boxes]
    keep = [i for i, c in enumerate(crops)
            if c is not None and min(c.shape[:2]) >= 4]
    if not keep:
        return out, embs
    e = vlm.encode_image([crops[i] for i in keep])
    p = vlm.similarity(e, BANK, softmax=True)[:, 0].cpu().numpy()
    idx = np.asarray(keep)
    out[idx] = p
    embs[idx] = e.cpu().numpy()
    return out, embs


cache, base_pi = [], []
t0 = time.time()
for i, iid in enumerate(test_ids):
    if i % 5 == 0:
        print(f'[det] {i}/{len(test_ids)}  {time.time()-t0:.0f}s', flush=True)
    img = det._img(iid)
    boxes, scores = det.predict_image(model, iid)
    gt = torch.as_tensor(det.image_boxes(iid), dtype=torch.float32).reshape(-1, 4)
    base_pi.append((gt, boxes, scores))
    b_np = boxes.cpu().numpy() if torch.is_tensor(boxes) else np.asarray(boxes)
    s_np = scores.cpu().numpy() if torch.is_tensor(scores) else np.asarray(scores)
    _v, _e = vlm_scores_for_boxes(img, b_np)
    cache.append((gt.numpy(), b_np.reshape(-1, 4), s_np.reshape(-1), _v, _e))

np.savez_compressed(os.path.join(OUT_DET, 'raw_detections.npz'),
                    **{f'{k}_{i}': v for i, c in enumerate(cache)
                       for k, v in zip(('gt', 'boxes', 'det', 'vlm', 'emb'), c)})

allv = np.concatenate([c[3] for c in cache]) if cache else np.zeros(1)
alld = np.concatenate([c[2] for c in cache]) if cache else np.zeros(1)
qs = (1, 5, 25, 50, 75, 95, 99)
stats = {'n_boxes': int(allv.size),
         'vlm_percentiles': {str(q): float(np.percentile(allv, q)) for q in qs},
         'det_percentiles': {str(q): float(np.percentile(alld, q)) for q in qs}}
print('[det] distributia scorurilor:', json.dumps(stats, indent=2), flush=True)

# --- diagnostic decisiv: e scorul VLM informativ pentru "box adevarat vs fals"? ---
from torchvision.ops import box_iou
from sklearn.metrics import roc_auc_score
_y, _v, _s, _a, _e = [], [], [], [], []
for gt_np, b_np, s_np, v_np, e_np in cache:
    if len(b_np) == 0:
        continue
    lab = (box_iou(torch.tensor(b_np), torch.tensor(gt_np)).numpy().max(1) >= det.FROC_IOU
           if len(gt_np) else np.zeros(len(b_np), bool))
    _y.append(lab); _v.append(v_np); _s.append(s_np); _e.append(e_np)
    _a.append((b_np[:, 2] - b_np[:, 0]) * (b_np[:, 3] - b_np[:, 1]))
diag = {}
Y_ALL = EMB_ALL = None
if _y:
    y, v, s, a = map(np.concatenate, (_y, _v, _s, _a))
    Y_ALL, EMB_ALL = y, np.concatenate(_e)
    if 0 < y.sum() < len(y):
        diag = {'n_boxes': int(len(y)), 'tp_rate': float(y.mean()),
                'auc_vlm': float(roc_auc_score(y, v)),
                'auc_detector': float(roc_auc_score(y, s)),
                'auc_arie_singura': float(roc_auc_score(y, -a)),
                'corr_vlm_latura': float(np.corrcoef(v, np.sqrt(a))[0, 1]),
                'vlm_mean_tp': float(v[y].mean()), 'vlm_mean_fp': float(v[~y].mean()),
                'latura_mediana_px': float(np.sqrt(np.median(a)))}
        print('[det] diagnostic separabilitate:', json.dumps(diag, indent=2), flush=True)
        print('[det] auc_vlm ~0.5 = neinformativ | <0.5 = anti-corelat | '
              'daca auc_arie_singura ~ auc_vlm inversat, e artefact de marime', flush=True)

_, base_sens = det.froc_boxes(base_pi)
print('[det] FROC baseline:', base_sens, flush=True)


# --- feedback #3/#8: mAP@IoU0.5 + Dice mediu (boxuri pastrate vs GT) -------
# FROC (sens@Xfp) raspunde "cat de sensibil e la un nr fix de FP/imagine".
# AP@50 raspunde "cat de buna e ordonarea scorurilor peste tot pragul" (metrica
# standard de detectie). Dice raspunde "cat de bine se suprapune box-ul cu GT",
# nu doar daca IoU>=prag (0/1) ca la FROC.
def average_precision_at_iou(pi, iou_thr=0.5):
    """AP (interpolare all-point, stil COCO) la un singur prag de IoU."""
    all_scores, all_tp, n_gt = [], [], 0
    for gt, boxes, scores in pi:
        n_gt += len(gt)
        if len(boxes) == 0:
            continue
        order = torch.argsort(scores, descending=True)
        bb, ss = boxes[order], scores[order]
        matched = torch.zeros(len(gt), dtype=torch.bool)
        ious = box_iou(bb, gt) if len(gt) else None
        for i in range(len(bb)):
            all_scores.append(float(ss[i]))
            if ious is None or len(gt) == 0:
                all_tp.append(0); continue
            j = int(torch.argmax(ious[i]))
            if ious[i, j] >= iou_thr and not matched[j]:
                all_tp.append(1); matched[j] = True
            else:
                all_tp.append(0)
    if n_gt == 0 or not all_scores:
        return 0.0
    order = np.argsort(-np.asarray(all_scores))
    tp = np.asarray(all_tp)[order]
    tp_cum, fp_cum = np.cumsum(tp), np.cumsum(1 - tp)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    ap, prev_r = 0.0, 0.0
    for p, r in zip(precision, recall):
        ap += p * (r - prev_r); prev_r = r
    return float(ap)


def mean_dice_boxes(pi):
    """Dice mediu intre fiecare box (pastrat) si cel mai apropiat GT (arie, nu doar IoU>=prag)."""
    dices = []
    for gt, boxes, _ in pi:
        if len(gt) == 0 or len(boxes) == 0:
            continue
        ious = box_iou(boxes, gt)
        for i in range(len(boxes)):
            j = int(torch.argmax(ious[i]))
            b, g = boxes[i], gt[j]
            ix1, iy1 = max(float(b[0]), float(g[0])), max(float(b[1]), float(g[1]))
            ix2, iy2 = min(float(b[2]), float(g[2])), min(float(b[3]), float(g[3]))
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
            area_g = max(0.0, float(g[2] - g[0])) * max(0.0, float(g[3] - g[1]))
            dices.append(2 * inter / (area_b + area_g) if (area_b + area_g) > 0 else 0.0)
    return float(np.mean(dices)) if dices else 0.0


ap50_baseline = average_precision_at_iou(base_pi, 0.5)
dice_baseline = mean_dice_boxes(base_pi)
print(f'[det] AP@50 baseline = {ap50_baseline:.4f}   Dice mediu baseline = {dice_baseline:.4f}', flush=True)

grid = []
for b in (0.0, 0.25, 0.5, 0.75):
    for mv in (0.0, 0.2, 0.35, 0.5, 0.65):
        pi, kept = [], 0
        for gt_np, b_np, s_np, v_np, _ in cache:
            fused, keep = rerank_and_filter(s_np, v_np, b, mv)
            kept += int(keep.sum())
            pi.append((torch.as_tensor(gt_np),
                       torch.as_tensor(b_np[keep], dtype=torch.float32),
                       torch.as_tensor(fused[keep], dtype=torch.float32)))
        _, sens = det.froc_boxes(pi)
        ap50 = average_precision_at_iou(pi, 0.5)
        dice_kept = mean_dice_boxes(pi)
        grid.append({'beta': b, 'min_vlm': mv, 'n_kept': kept, 'froc': sens,
                     'ap50': ap50, 'dice_boxes': dice_kept})
        print(f'[det] beta={b} min_vlm={mv} kept={kept} AP50={ap50:.3f} '
              f'dice={dice_kept:.3f} -> {sens}', flush=True)

det_summary = {'smoke': SMOKE, 'fold': FOLD, 'n_images': len(test_ids),
               'min_crop': MIN_CROP, 'baseline': base_sens, 'score_stats': stats,
               'diagnostic': diag, 'grid': grid,
               'ap50_baseline': ap50_baseline, 'dice_boxes_baseline': dice_baseline,
               'minutes': round((time.time() - t0) / 60, 1)}
with open(os.path.join(OUT_DET, 'detection_vlm_summary.json'), 'w') as f:
    json.dump(det_summary, f, indent=2, default=float)
print('gata detectie ->', OUT_DET)'''))

cells.append(md("""## Ploturi — curbe FROC, grila beta/min_vlm, distributia scorurilor

Nu doar cifrele din grila text: curbe FROC baseline vs VLM pentru cateva praguri,
heatmap sens@1fp pe toata grila `beta` x `min_vlm`, histograma scorurilor VLM
pentru boxuri adevarate vs false, si sensibilitatea in functie de prag."""))

cells.append(code('''# ===== PLOTURI — detectie: curbe FROC, grila beta/min_vlm, distributie scoruri =====
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re


def _fp_key_to_x(k):
    return float(re.search(r'([\\d.]+)fp', k).group(1))


fp_keys = sorted(base_sens.keys(), key=_fp_key_to_x)
fp_x = [_fp_key_to_x(k) for k in fp_keys]

fig, ax = plt.subplots(2, 2, figsize=(13, 10))

# 1) curbe FROC: baseline vs cateva configuratii min_vlm (beta=0)
ax[0, 0].plot(fp_x, [base_sens[k] for k in fp_keys], 'o-', color='#888888',
             label='baseline (identitate)', linewidth=2)
_palette = ['#4c72b0', '#55a868', '#c44e52', '#8172b2']
_shown = [g for g in grid if g['beta'] == 0.0 and g['min_vlm'] in (0.2, 0.35, 0.5)]
for g, col in zip(_shown, _palette):
    ax[0, 0].plot(fp_x, [g['froc'][k] for k in fp_keys], 'o--', color=col,
                 label=f"beta=0 min_vlm={g['min_vlm']}")
ax[0, 0].set_xlabel('fals-pozitive / imagine'); ax[0, 0].set_ylabel('sensibilitate')
ax[0, 0].set_title('Curbe FROC: baseline vs reranking VLM')
ax[0, 0].legend(fontsize=8); ax[0, 0].grid(alpha=0.3)

# 2) heatmap sens@1fp pe grila beta x min_vlm
betas = sorted({g['beta'] for g in grid})
min_vlms = sorted({g['min_vlm'] for g in grid})
key1fp = min(fp_keys, key=lambda k: abs(_fp_key_to_x(k) - 1.0))
H = np.zeros((len(betas), len(min_vlms)))
for g in grid:
    H[betas.index(g['beta']), min_vlms.index(g['min_vlm'])] = g['froc'].get(key1fp, 0.0)
im = ax[0, 1].imshow(H, cmap='viridis', aspect='auto', vmin=0, vmax=1)
ax[0, 1].set_xticks(range(len(min_vlms))); ax[0, 1].set_xticklabels(min_vlms)
ax[0, 1].set_yticks(range(len(betas))); ax[0, 1].set_yticklabels(betas)
ax[0, 1].set_xlabel('min_vlm'); ax[0, 1].set_ylabel('beta')
ax[0, 1].set_title(f'{key1fp} pe grila beta x min_vlm')
for i in range(len(betas)):
    for j in range(len(min_vlms)):
        ax[0, 1].text(j, i, f'{H[i, j]:.2f}', ha='center', va='center',
                     color='white' if H[i, j] < 0.5 else 'black', fontsize=8)
plt.colorbar(im, ax=ax[0, 1], fraction=0.046)

# 3) distributia scorurilor VLM: TP vs FP
if _y:
    bins = np.linspace(0, 1, 26)
    ax[1, 0].hist(v[y], bins=bins, alpha=0.6, label=f'box adevarat (n={int(y.sum())})', color='#55a868')
    ax[1, 0].hist(v[~y], bins=bins, alpha=0.6, label=f'box fals (n={int((~y).sum())})', color='#c44e52')
    ax[1, 0].axvline(0.2, color='k', linestyle='--', linewidth=1, label='prag min_vlm=0.2')
    ax[1, 0].set_xlabel('scor VLM'); ax[1, 0].set_ylabel('numar boxuri')
    ax[1, 0].set_title(f"scor VLM: TP vs FP (AUC={diag.get('auc_vlm', float('nan')):.3f})")
    ax[1, 0].legend(fontsize=8)
else:
    ax[1, 0].axis('off')

# 4) sens@1fp in functie de min_vlm, o linie per beta
for b, col in zip(betas, _palette + ['#333333']):
    row = sorted([g for g in grid if g['beta'] == b], key=lambda g: g['min_vlm'])
    ax[1, 1].plot([g['min_vlm'] for g in row], [g['froc'].get(key1fp, 0.0) for g in row],
                 'o-', color=col, label=f'beta={b}')
ax[1, 1].axhline(base_sens.get(key1fp, 0.0), color='#888888', linestyle='--', label='baseline')
ax[1, 1].set_xlabel('min_vlm (prag filtrare)'); ax[1, 1].set_ylabel(f'sens@{key1fp}')
ax[1, 1].set_title('Sensibilitate vs prag de filtrare VLM')
ax[1, 1].legend(fontsize=8); ax[1, 1].grid(alpha=0.3)

_best_g = min(grid, key=lambda g: abs(g['beta'] - 0.0) + abs(g['min_vlm'] - 0.2))
fig.suptitle(
    f"AP@50: baseline={ap50_baseline:.3f}  vs  beta=0,min_vlm=0.2={_best_g['ap50']:.3f}   |   "
    f"Dice mediu boxuri: baseline={dice_baseline:.3f}  vs  beta=0,min_vlm=0.2={_best_g['dice_boxes']:.3f}",
    fontsize=10)
plt.tight_layout(rect=[0, 0, 1, 0.95])
_p = os.path.join(OUT_DET, 'detection_summary_plots.png')
plt.savefig(_p, dpi=130)
plt.close(fig)
print('[plot] salvat', _p, flush=True)'''))

cells.append(md("""## Vizualizari calitative — detectie

Salvez imagini intregi cu box-urile GT (verde), box-urile pastrate de VLM
(min_vlm=0.2, albastru) si box-urile eliminate de VLM (rosu) — ca sa vad daca
filtrarea scoate zgomot sau leziuni adevarate, nu doar cifra de sensibilitate.
Fiecare box e etichetat cu scorul VLM si TP/FP (fata de GT), ca sa fie clar
DE CE a fost pastrat sau eliminat, nu doar culoarea.

**Selectia** (ca la segmentare) nu mai e uniforma: aleg explicit imaginile
"bune" (VLM elimina zgomot, nu atinge leziuni adevarate) si "proaste" (VLM
elimina o leziune adevarata) dupa cate TP-uri, respectiv FP-uri, elimina
filtrarea `min_vlm=0.2` — plus cateva medii pentru context."""))

cells.append(code('''# ===== VIZUALIZARI: DETECTIE (exemple BUNE/PROASTE, nu esantion uniform) =====
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision.ops import box_iou as _box_iou_viz

VIZ_DET_DIR = '/kaggle/working/qualitative/detectie'
os.makedirs(VIZ_DET_DIR, exist_ok=True)

_MIN_VLM_VIZ = 0.2
_prompt_caption_det = ('prompturi pozitive: ' + '; '.join(DETECTION_POS_PROMPTS) +
                       '  |  prompturi negative: ' + '; '.join(DET_NEG_PROMPTS))


def _tp_labels(gt_np, b_np):
    """True/False per box: are IoU >= FROC_IOU cu vreun GT?"""
    if len(gt_np) == 0 or len(b_np) == 0:
        return np.zeros(len(b_np), bool)
    ious = _box_iou_viz(torch.as_tensor(b_np, dtype=torch.float32),
                        torch.as_tensor(gt_np, dtype=torch.float32)).numpy()
    return ious.max(1) >= det.FROC_IOU


# pentru fiecare imagine: cate TP-uri, respectiv FP-uri, elimina min_vlm=0.2
_per_img = []
for i, (gt_np, b_np, s_np, v_np, _) in enumerate(cache):
    if len(b_np) == 0:
        _per_img.append((i, 0, 0)); continue
    tp = _tp_labels(gt_np, b_np)
    removed = v_np < _MIN_VLM_VIZ
    _per_img.append((i, int((removed & tp).sum()), int((removed & ~tp).sum())))

_bad_order = sorted(_per_img, key=lambda t: -t[1])     # cele mai multe TP-uri pierdute
_good_order = sorted(_per_img, key=lambda t: -t[2])    # cele mai multe FP-uri curatate (fara TP pierdut)
_good_order = [t for t in _good_order if t[1] == 0] or _good_order

N_VIZ_DET = min(12, len(cache))
_n_bad = min(4, N_VIZ_DET)
_n_good = min(4, N_VIZ_DET)
_bad_idx = [t[0] for t in _bad_order[:_n_bad] if t[1] > 0]
_good_idx = [t[0] for t in _good_order[:_n_good] if t[0] not in _bad_idx]
_used = set(_bad_idx) | set(_good_idx)
_rest = max(0, N_VIZ_DET - len(_used))
_mid_pool = [t[0] for t in _per_img if t[0] not in _used]
_mid_idx = list(np.linspace(0, len(_mid_pool) - 1, _rest).astype(int)) if _mid_pool and _rest > 0 else []
_mid_idx = [_mid_pool[j] for j in _mid_idx]
_viz_idx_det = _bad_idx + _mid_idx + _good_idx
_viz_cat_det = (['prost'] * len(_bad_idx) + ['mediu'] * len(_mid_idx) + ['bun'] * len(_good_idx))

# det_viz_records tine, per imagine PNG salvata, numele imaginii (iid) + cate boxuri
# GT/propuse/pastrate/eliminate + categoria -> intra in documentul de evaluare.
det_viz_records = []
for k, (i, cat) in enumerate(zip(_viz_idx_det, _viz_cat_det)):
    iid = test_ids[i]
    gt_np, b_np, s_np, v_np, _ = cache[i]
    img = det._img(iid)
    keep = v_np >= _MIN_VLM_VIZ
    tp = _tp_labels(gt_np, b_np)

    # feedback: pe imagini aglomerate (100+ boxuri), o eticheta per box devine
    # ilizibila (text suprapus). Etichetez toate boxurile doar sub un prag; peste,
    # etichetez DOAR TP-urile (putine, cele mai importante clinic -- arata exact
    # ce leziuni reale s-au pierdut), fara sa mai aglomerez zona de FP-uri dense.
    _n_boxes_img = len(b_np)
    _label_all = _n_boxes_img <= 40
    _figsize = (8, 8) if _n_boxes_img <= 40 else (11, 11)
    fig, ax = plt.subplots(figsize=_figsize)
    ax.imshow(img, cmap='gray')
    for x1, y1, x2, y2 in gt_np:
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='lime', linewidth=2))
    for (x1, y1, x2, y2), kp, vs, is_tp in zip(b_np, keep, v_np, tp):
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                   edgecolor=('deepskyblue' if kp else 'red'), linewidth=1))
        if _label_all or is_tp:
            ax.text(x1, max(0, y1 - 3), f'v={vs:.2f} {"TP" if is_tp else "FP"}',
                   color=('deepskyblue' if kp else 'red'), fontsize=6)
    _extra = '' if _label_all else f'  (doar TP-urile etichetate, {_n_boxes_img} boxuri total)'
    ax.set_title(f'{iid} [{cat.upper()}]  verde=GT  albastru=pastrat(VLM>={_MIN_VLM_VIZ})  rosu=eliminat  '
                f'(TP pierdute={int((~keep & tp).sum())}, FP curatate={int((~keep & ~tp).sum())}){_extra}', fontsize=9)
    fig.text(0.5, 0.01, _prompt_caption_det, ha='center', fontsize=7)
    ax.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    fname = f'{k:02d}_{cat}_{iid}.png'
    plt.savefig(os.path.join(VIZ_DET_DIR, fname), dpi=110)
    plt.close(fig)
    det_viz_records.append({'idx': k, 'category': cat, 'image_id': str(iid), 'file': fname,
                            'n_gt_boxes': int(len(gt_np)), 'n_proposed_boxes': int(len(b_np)),
                            'n_kept_vlm': int(keep.sum()), 'n_removed_vlm': int((~keep).sum()),
                            'n_tp_removed': int((~keep & tp).sum()), 'n_fp_removed': int((~keep & ~tp).sum()),
                            'prompts_pos': list(DETECTION_POS_PROMPTS), 'prompts_neg': list(DET_NEG_PROMPTS)})

with open(os.path.join(VIZ_DET_DIR, 'index.json'), 'w') as f:
    json.dump(det_viz_records, f, indent=2)
print(f'[viz] {len(_viz_idx_det)} imagini salvate in {VIZ_DET_DIR} '
      f'({len(_bad_idx)} proaste, {len(_mid_idx)} medii, {len(_good_idx)} bune)', flush=True)'''))

cells.append(md("""## Ablatie de prompturi (gratuita)

Embedding-urile crop-urilor sunt deja calculate si pastrate, deci putem testa
oricate seturi de prompturi fara sa mai atingem imaginile. Se re-encodeaza doar
textul (milisecunde). Metrica: AUC "box adevarat vs fals" — **0.5 = zero informatie**,
iar cat de departe e de 0.5 conteaza mai mult decat directia."""))

cells.append(code('''# ===== ABLATIE PROMPTURI =====
import numpy as np, torch, torch.nn.functional as Fnn
from sklearn.metrics import roc_auc_score
assert EMB_ALL is not None, 'Ruleaza intai celula de detectie.'

E = Fnn.normalize(torch.tensor(EMB_ALL), dim=-1)
valid = EMB_ALL.any(1)
ls = float(vlm.logit_scale)

VARIANTE = {
    'original': (DETECTION_POS_PROMPTS, DETECTION_NEG_PROMPTS),
    'fara_artifact': (DETECTION_POS_PROMPTS,
                      ['normal breast tissue without calcifications',
                       'empty background outside the breast']),
    'neg_doar_tesut': (DETECTION_POS_PROMPTS,
                       ['normal fibroglandular breast tissue']),
    'pos_simplu': (['calcifications in the breast'],
                   ['breast tissue without calcifications']),
    'pos_bright': (['small bright white calcium deposits in breast tissue',
                    'a cluster of tiny bright specks on a mammogram'],
                   ['uniform breast tissue with no bright specks']),
}

rows = []
for name, (pp, nn) in VARIANTE.items():
    with torch.no_grad():
        pos = Fnn.normalize(vlm.encode_text(list(pp)).mean(0, keepdim=True), dim=-1)
        neg = Fnn.normalize(vlm.encode_text(list(nn)).mean(0, keepdim=True), dim=-1)
        bank = Fnn.normalize(torch.cat([pos, neg], 0), dim=-1).cpu()
    sc = torch.softmax(ls * (E @ bank.t()), dim=-1)[:, 0].numpy()
    auc = float(roc_auc_score(Y_ALL[valid], sc[valid]))
    rows.append({'set': name, 'auc': auc, 'scor_mediu': float(sc[valid].mean()),
                 'sd': float(sc[valid].std())})
    print(f'{name:16s} AUC={auc:.3f}  scor mediu={sc[valid].mean():.3f}  sd={sc[valid].std():.3f}')

best = max(rows, key=lambda r: abs(r['auc'] - 0.5))
print(f"\\nCel mai informativ: {best['set']}  AUC={best['auc']:.3f}  "
      f"(distanta fata de 0.5 = {abs(best['auc']-0.5):.3f})")
with open(os.path.join(OUT_DET, 'prompt_ablation.json'), 'w') as f:
    json.dump(rows, f, indent=2)'''))

cells.append(md("""## Plot — ablatie prompturi

AUC si scor mediu per set de prompturi, unul langa altul — mai usor de comparat
cu ochiul decat coloana de numere de mai sus."""))

cells.append(code('''# ===== PLOT — ablatie prompturi =====
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
names = [r['set'] for r in rows]
aucs = [r['auc'] for r in rows]
cols = ['#c44e52' if n == best['set'] else '#4c72b0' for n in names]
ax[0].barh(names, aucs, color=cols)
ax[0].axvline(0.5, color='k', linestyle='--', linewidth=1, label='0.5 = neinformativ')
for i, a in enumerate(aucs):
    ax[0].text(a + 0.01, i, f'{a:.3f}', va='center', fontsize=9)
ax[0].set_xlabel('AUC (box adevarat vs fals)')
ax[0].set_title('Ablatie prompturi — cat de informativ e fiecare set')
ax[0].legend()

means = [r['scor_mediu'] for r in rows]
sds = [r['sd'] for r in rows]
ax[1].barh(names, means, xerr=sds, color=cols, capsize=4)
ax[1].set_xlabel('scor VLM mediu ± sd')
ax[1].set_title('Scor mediu per set de prompturi')

plt.tight_layout()
_p = os.path.join(OUT_DET, 'prompt_ablation_plot.png')
plt.savefig(_p, dpi=130)
plt.close(fig)
print('[plot] salvat', _p, flush=True)'''))

cells.append(md("""## Generalizare — acelasi test pe mai multe fold-uri

Rezultatele de detectie de mai sus sunt pe **fold 1**. Ca sa verific ca nu e un
rezultat particular acelui fold, repet evaluarea baseline vs. VLM (configuratia
castigatoare `beta=0, min_vlm=0.2`) pe toate fold-urile din `folds.csv` si
raportez media +/- deviatia standard intre fold-uri, plus in cate fold-uri VLM
chiar bate baseline-ul (nu doar in medie)."""))

cells.append(code('''# ===== GENERALIZARE: DETECTIE PE MAI MULTE FOLD-URI =====
import os, json, time, numpy as np, torch, pandas as pd

_fdf = pd.read_csv(FOLDS_CSV)
ALL_FOLDS = sorted(int(c.split('_')[1]) for c in _fdf.columns if c.startswith('fold_'))
if SMOKE:
    ALL_FOLDS = ALL_FOLDS[:2]   # smoke: doar 2 fold-uri, N_DET imagini fiecare

BETA_G, MIN_VLM_G = 0.0, 0.2   # configuratia castigatoare pe fold 1 (sens@1fp)


def eval_fold(fold_id):
    _, _, tids = det.load_fold(fold_id)
    if N_DET:
        tids = tids[:N_DET]
    cache_f, base_pi_f = [], []
    for iid in tids:
        img = det._img(iid)
        boxes, scores = det.predict_image(model, iid)
        gt = torch.as_tensor(det.image_boxes(iid), dtype=torch.float32).reshape(-1, 4)
        base_pi_f.append((gt, boxes, scores))
        b_np = boxes.cpu().numpy() if torch.is_tensor(boxes) else np.asarray(boxes)
        s_np = scores.cpu().numpy() if torch.is_tensor(scores) else np.asarray(scores)
        v_np, _ = vlm_scores_for_boxes(img, b_np)
        cache_f.append((gt.numpy(), b_np.reshape(-1, 4), s_np.reshape(-1), v_np))
    _, base_sens_f = det.froc_boxes(base_pi_f)
    pi = []
    for gt_np, b_np, s_np, v_np in cache_f:
        fused, keep = rerank_and_filter(s_np, v_np, BETA_G, MIN_VLM_G)
        pi.append((torch.as_tensor(gt_np), torch.as_tensor(b_np[keep], dtype=torch.float32),
                   torch.as_tensor(fused[keep], dtype=torch.float32)))
    _, vlm_sens_f = det.froc_boxes(pi)
    return {'fold': int(fold_id), 'n_images': len(tids), 'baseline': base_sens_f, 'vlm': vlm_sens_f}


t0 = time.time()
multi = []
for fid in ALL_FOLDS:
    print(f'[gen] fold {fid} ...', flush=True)
    r = eval_fold(fid)
    multi.append(r)
    print(f'[gen] fold {fid}: baseline={r["baseline"]}  vlm={r["vlm"]}', flush=True)

keys = sorted({k for r in multi for k in r['baseline'].keys()})
gen_summary = {'beta': BETA_G, 'min_vlm': MIN_VLM_G, 'folds': multi,
               'minutes': round((time.time() - t0) / 60, 1), 'per_metric': {}}
for k in keys:
    base_vals = [r['baseline'][k] for r in multi if k in r['baseline']]
    vlm_vals = [r['vlm'][k] for r in multi if k in r['vlm']]
    if base_vals and vlm_vals:
        wins = sum(v > b for v, b in zip(vlm_vals, base_vals))
        gen_summary['per_metric'][k] = {
            'baseline_mean': float(np.mean(base_vals)), 'baseline_std': float(np.std(base_vals)),
            'vlm_mean': float(np.mean(vlm_vals)), 'vlm_std': float(np.std(vlm_vals)),
            'n_folds_vlm_better': wins, 'n_folds': len(vlm_vals)}
        print(f'{k}: baseline {np.mean(base_vals):.3f}+/-{np.std(base_vals):.3f}  '
              f'vlm {np.mean(vlm_vals):.3f}+/-{np.std(vlm_vals):.3f}  '
              f'(VLM mai bun in {wins}/{len(vlm_vals)} fold-uri)', flush=True)

with open(os.path.join(OUT_DET, 'detection_multifold_summary.json'), 'w') as f:
    json.dump(gen_summary, f, indent=2, default=float)
print('gata generalizare ->', OUT_DET)'''))

cells.append(md("""## Plot — generalizare multi-fold

Media +/- std ascunde varianta reala intre fold-uri. Aici: bar-uri baseline vs
VLM per metrica (cu bare de eroare), plus traiectoria sens@1fp fold-cu-fold."""))

cells.append(code('''# ===== PLOT — generalizare multi-fold =====
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re


def _fp_key_to_x(k):
    return float(re.search(r'([\\d.]+)fp', k).group(1))


metrics = sorted(gen_summary['per_metric'].keys(), key=_fp_key_to_x)
fig, ax = plt.subplots(1, 2, figsize=(13, 5))

# 1) bar mediu +/- std, baseline vs VLM, per metrica
x = np.arange(len(metrics))
w = 0.35
base_m = [gen_summary['per_metric'][k]['baseline_mean'] for k in metrics]
base_s = [gen_summary['per_metric'][k]['baseline_std'] for k in metrics]
vlm_m = [gen_summary['per_metric'][k]['vlm_mean'] for k in metrics]
vlm_s = [gen_summary['per_metric'][k]['vlm_std'] for k in metrics]
ax[0].bar(x - w / 2, base_m, w, yerr=base_s, capsize=4, label='baseline', color='#888888')
ax[0].bar(x + w / 2, vlm_m, w, yerr=vlm_s, capsize=4, label='VLM (beta=0, min_vlm=0.2)', color='#4c72b0')
ax[0].set_xticks(x); ax[0].set_xticklabels([k.replace('sens@', '') for k in metrics], rotation=0)
ax[0].set_ylabel('sensibilitate'); ax[0].set_title(f'Baseline vs VLM, medie ± std pe {len(multi)} fold-uri')
ax[0].legend(fontsize=8)

# 2) traiectorie per fold pentru fp-ul cel mai apropiat de 1.0
key1fp = min(metrics, key=lambda k: abs(_fp_key_to_x(k) - 1.0))
folds_x = [r['fold'] for r in multi]
ax[1].plot(folds_x, [r['baseline'].get(key1fp, 0.0) for r in multi], 'o--', color='#888888', label='baseline')
ax[1].plot(folds_x, [r['vlm'].get(key1fp, 0.0) for r in multi], 'o-', color='#4c72b0', label='VLM')
ax[1].set_xlabel('fold'); ax[1].set_ylabel(f'sens@{key1fp}')
ax[1].set_title(f'sens@{key1fp} pe fiecare fold (nu doar media)')
ax[1].set_xticks(folds_x)
ax[1].legend(fontsize=8)

plt.tight_layout()
_p = os.path.join(OUT_DET, 'multifold_summary_plot.png')
plt.savefig(_p, dpi=130)
plt.close(fig)
print('[plot] salvat', _p, flush=True)'''))

cells.append(md("""## Document de evaluare (cantitativ + calitativ, organizat)

Feedback primit: evaluarile (cantitative si calitative) sa fie organizate intr-un
document usor de urmarit, iar la imaginile analizate calitativ sa apara si numele
lor, ca sa se poata compara ulterior intre metode pe **aceeasi** imagine.

Celula de mai jos aduna tot ce s-a calculat pana acum (JSON-urile de rezumat +
`index.json`-urile din `qualitative/`) intr-un singur fisier Markdown,
`/kaggle/working/EVALUARE_TEMA2.md`: tabele cantitative per etapa, apoi cate un
tabel calitativ cu `image_id`/`patch` + fisierul PNG + metricile pe acea imagine."""))

cells.append(code('''# ===== DOCUMENT DE EVALUARE (cantitativ + calitativ, organizat) =====
import json, os


def _load(p):
    try:
        return json.load(open(p))
    except FileNotFoundError:
        return None


REPORT_PATH = '/kaggle/working/EVALUARE_TEMA2.md'
L = []
L.append('# Evaluare Tema 2 -- VLM peste detectie si segmentare\\n')
L.append(f'_Fold={FOLD}  SMOKE={SMOKE}_\\n')

_cls = _load('/kaggle/working/VLM_Classification_Results/summary.json')
if _cls:
    L.append('\\n## 0. Clasificare -- evaluare cantitativa\\n')
    L.append('| varianta | acuratete |\\n|---|---|\\n')
    for k in ['cnn_only_raw', 'vlm_zeroshot', 'cnn_trained', 'vlm_trained', 'feature_fusion']:
        if k in _cls:
            L.append(f'| {k} | {_cls[k]:.4f} |\\n')
    if 'feature_fusion' in _cls and 'cnn_trained' in _cls:
        L.append(f"\\n**Contributie marginala VLM (fusion - cnn_trained) = "
                 f"{_cls['feature_fusion'] - _cls['cnn_trained']:+.4f}**\\n")
    if 'per_class_f1' in _cls:
        L.append('\\n### De unde vine diferenta feature_fusion (F1 per clasa)\\n\\n')
        _cnames = list(next(iter(_cls['per_class_f1'].values())).keys())
        L.append('| varianta | ' + ' | '.join(_cnames) + ' |\\n')
        L.append('|---|' + '---|' * len(_cnames) + '\\n')
        for _v in ('cnn_trained', 'vlm_trained', 'feature_fusion'):
            if _v in _cls['per_class_f1']:
                L.append(f'| {_v} | ' + ' | '.join(f"{_cls['per_class_f1'][_v][c]:.3f}" for c in _cnames) + ' |\\n')
    if 'case_breakdown' in _cls:
        L.append('\\n### Analiza pe cazuri (cate patch-uri castiga/pierde fuziunea)\\n\\n')
        L.append('| caz | numar patch-uri |\\n|---|---|\\n')
        for kk, vv in _cls['case_breakdown'].items():
            if kk != 'n_total_val':
                L.append(f'| {kk} | {vv} |\\n')

L.append('\\n## 1. Detectie -- evaluare cantitativa\\n')
L.append(f"Baseline (identitate), {det_summary['n_images']} imagini, fold {FOLD}:\\n\\n")
L.append(f"**AP@IoU0.5 baseline = {det_summary['ap50_baseline']:.4f}**   "
         f"**Dice mediu boxuri baseline = {det_summary['dice_boxes_baseline']:.4f}**\\n\\n")
L.append('| prag FP/imagine | sensibilitate baseline |\\n|---|---|\\n')
for kk, v in base_sens.items():
    L.append(f'| {kk} | {v:.3f} |\\n')

L.append('\\n### Grila reranking VLM (beta, min_vlm)\\n\\n')
L.append('| beta | min_vlm | boxuri pastrate | AP@50 | Dice boxuri | ' +
         ' | '.join(base_sens.keys()) + ' |\\n')
L.append('|---|---|---|---|---|' + '---|' * len(base_sens) + '\\n')
for g in det_summary['grid']:
    L.append(f"| {g['beta']} | {g['min_vlm']} | {g['n_kept']} | {g['ap50']:.3f} | {g['dice_boxes']:.3f} | " +
             ' | '.join(f"{g['froc'].get(kk, 0):.3f}" for kk in base_sens.keys()) + ' |\\n')

if det_summary.get('diagnostic'):
    L.append('\\n### Diagnostic separabilitate (box adevarat vs fals)\\n\\n')
    for kk, v in det_summary['diagnostic'].items():
        L.append(f'- **{kk}** = {v}\\n')

L.append('\\n### Ablatie prompturi\\n\\n')
L.append('| set prompturi | AUC | scor mediu | sd |\\n|---|---|---|---|\\n')
for r in rows:
    L.append(f"| {r['set']} | {r['auc']:.3f} | {r['scor_mediu']:.3f} | {r['sd']:.3f} |\\n")

_gm = _load(f'{OUT_DET}/detection_multifold_summary.json')
if _gm:
    L.append(f"\\n### Generalizare multi-fold (beta={_gm['beta']}, min_vlm={_gm['min_vlm']})\\n\\n")
    L.append('| metrica | baseline (medie+/-std) | VLM (medie+/-std) | VLM mai bun in |\\n|---|---|---|---|\\n')
    for kk, v in _gm['per_metric'].items():
        L.append(f"| {kk} | {v['baseline_mean']:.3f}+/-{v['baseline_std']:.3f} | "
                 f"{v['vlm_mean']:.3f}+/-{v['vlm_std']:.3f} | {v['n_folds_vlm_better']}/{v['n_folds']} |\\n")

L.append('\\n## 2. Detectie -- evaluare calitativa (imagini inspectate)\\n\\n')
L.append('Imagini cu box-uri GT (verde) / pastrate de VLM (albastru) / eliminate (rosu), '
         'fiecare box etichetat cu scorul VLM si TP/FP. Selectate explicit ca exemple '
         '**bune** (VLM curata FP-uri fara sa piarda TP-uri) si **proaste** (VLM elimina '
         'o leziune adevarata), nu esantion aleatoriu -- ca sa se vada DE CE s-a pastrat '
         'sau eliminat fiecare box. Prompturile text folosite apar pe fiecare figura. '
         'Fisiere in `qualitative/detectie/`.\\n\\n')
L.append('| # | categorie | imagine (iid) | fisier | boxuri GT | boxuri propuse | '
         'pastrate VLM | eliminate VLM | TP pierdute | FP curatate |\\n')
L.append('|---|---|---|---|---|---|---|---|---|---|\\n')
for r in det_viz_records:
    L.append(f"| {r['idx']} | {r['category']} | {r['image_id']} | {r['file']} | {r['n_gt_boxes']} | "
             f"{r['n_proposed_boxes']} | {r['n_kept_vlm']} | {r['n_removed_vlm']} | "
             f"{r['n_tp_removed']} | {r['n_fp_removed']} |\\n")
_bad_det = [r for r in det_viz_records if r['category'] == 'prost']
_good_det = [r for r in det_viz_records if r['category'] == 'bun']
if _bad_det:
    _items = ', '.join(f"{r['image_id']} ({r['file']}, {r['n_tp_removed']} TP pierdute)" for r in _bad_det)
    L.append('\\n**Exemple proaste** (VLM a eliminat leziuni adevarate): ' + _items + '\\n')
if _good_det:
    _items = ', '.join(f"{r['image_id']} ({r['file']}, {r['n_fp_removed']} FP curatate)" for r in _good_det)
    L.append('\\n**Exemple bune** (VLM a curatat FP fara sa piarda TP-uri): ' + _items + '\\n')

L.append('\\n## 3. Segmentare -- evaluare cantitativa\\n\\n')
L.append('| metrica | valoare |\\n|---|---|\\n')
for kk in ['n_patches', 'n_vlm_prompted', 'n_det_prompted', 'dice_gt_box', 'dice_vlm_box',
          'dice_det_box', 'mean_mask_relevance_iou', 'mean_vlm_box_area_frac', 'mean_gt_box_area_frac']:
    L.append(f'| {kk} | {seg_summary.get(kk)} |\\n')

L.append('\\n## 4. Segmentare -- evaluare calitativa (patch-uri inspectate)\\n\\n')
L.append('Patch + masca GT + harta de relevanta VLM + masca MedSAM(box VLM) + masca MedSAM(box detectie). '
         'Selectate explicit dupa `dice_det_box`: cele mai **proaste** si cele mai **bune** cazuri, '
         'plus cateva medii -- nu esantion aleatoriu. Prompturile text apar pe fiecare figura. '
         'Fisiere in `qualitative/segmentare/`.\\n\\n')
L.append('| # | categorie | imagine (iid) | patch (x,y) | fisier | dice_gt_box | dice_vlm_box | dice_det_box |\\n')
L.append('|---|---|---|---|---|---|---|---|\\n')
for r in seg_viz_records:
    L.append(f"| {r['idx']} | {r['category']} | {r['image_id']} | {tuple(r['patch_xy'])} | {r['file']} | "
             f"{r['dice_gt_box']} | {r['dice_vlm_box']} | {r['dice_det_box']} |\\n")
_bad_seg = [r for r in seg_viz_records if r['category'] == 'prost']
_good_seg = [r for r in seg_viz_records if r['category'] == 'bun']
if _bad_seg:
    _items = ', '.join(f"{r['image_id']} ({r['file']}, dice_det={r['dice_det_box']})" for r in _bad_seg)
    L.append('\\n**Exemple proaste** (dice_det_box mic -- localizare rea): ' + _items + '\\n')
if _good_seg:
    _items = ', '.join(f"{r['image_id']} ({r['file']}, dice_det={r['dice_det_box']})" for r in _good_seg)
    L.append('\\n**Exemple bune** (dice_det_box mare -- localizare buna): ' + _items + '\\n')

L.append('\\n---\\n_Nota: acelasi `image_id` (iid) apare si in tabelul de detectie si in cel de '
         'segmentare -> rezultatele celor doua metode se pot compara direct pe aceeasi imagine._\\n')

with open(REPORT_PATH, 'w', encoding='utf-8') as f:
    f.writelines(L)
print('[raport] scris', REPORT_PATH, os.path.getsize(REPORT_PATH), 'bytes', flush=True)'''))

cells.append(code('''# ===== SUMAR FINAL =====
import json
print('===== SEGMENTARE =====')
print(json.dumps(json.load(open(f'{OUT_SEG}/segmentation_vlm_summary.json')), indent=2))
print('\\n===== DETECTIE (fold unic) =====')
d = json.load(open(f'{OUT_DET}/detection_vlm_summary.json'))
print('baseline:', json.dumps(d['baseline'], indent=2))
print(f"AP@50 baseline = {d['ap50_baseline']:.4f}   Dice mediu boxuri baseline = {d['dice_boxes_baseline']:.4f}")
print('distributie scoruri VLM:', json.dumps(d['score_stats']['vlm_percentiles'], indent=2))
print('\\ngrila beta/min_vlm:')
for g in d['grid']:
    print(f"  beta={g['beta']:<5} min_vlm={g['min_vlm']:<5} kept={g['n_kept']:<6} "
          f"AP50={g['ap50']:.3f} dice={g['dice_boxes']:.3f} {g['froc']}")
try:
    print('\\n===== GENERALIZARE (mai multe fold-uri) =====')
    print(json.dumps(json.load(open(f'{OUT_DET}/detection_multifold_summary.json'))['per_metric'], indent=2))
except FileNotFoundError:
    pass
print('\\nDocument de evaluare (cantitativ + calitativ, cu nume de imagini): '
      '/kaggle/working/EVALUARE_TEMA2.md')
print('Imagini de inspectat cu ochiul: /kaggle/working/qualitative/{segmentare,detectie}/')
print('Ploturi agregate:')
print(' ', os.path.join(OUT_SEG, 'segmentation_summary_plots.png'))
print(' ', os.path.join(OUT_DET, 'detection_summary_plots.png'))
print(' ', os.path.join(OUT_DET, 'prompt_ablation_plot.png'))
print(' ', os.path.join(OUT_DET, 'multifold_summary_plot.png'))'''))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}

out = r"C:\Users\rauce\Desktop\practica\BreastCancer\vlm\Tema2_Etape23_Kaggle.ipynb"
with io.open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=True)
print("scris:", out, os.path.getsize(out), "bytes")

# validare: json valid + fiecare celula de cod compileaza
nb2 = json.load(io.open(out, encoding="utf-8"))
import ast
n_ok = 0
for i, c in enumerate(nb2["cells"]):
    if c["cell_type"] == "code":
        src = c["source"] if isinstance(c["source"], str) else "".join(c["source"])
        src = "\n".join(l for l in src.split("\n") if not l.strip().startswith("!"))
        ast.parse(src)
        n_ok += 1
print("celule de cod validate sintactic:", n_ok, "/ total celule:", len(nb2["cells"]))
