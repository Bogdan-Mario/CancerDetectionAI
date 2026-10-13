"""
Download Mammo-CLIP pretrained checkpoints (shawn24/Mammo-CLIP)
    python download_mammo_clip.py            # b5 (default)
    python download_mammo_clip.py b2         # or:  b5 / b2 / both
    python download_mammo_clip.py list       # list all files in the repo (if a name is wrong)

"""
import os
import sys
import shutil

REPO_ID   = "shawn24/Mammo-CLIP"
SUBFOLDER = "Pre-trained-checkpoints"
FILES = {
    "b5": "b5-model-best-epoch-7.tar",
    "b2": "b2-model-best-epoch-10.tar",
}
OUT_DIR = "checkpoints"


def download_hub(which):
    from huggingface_hub import hf_hub_download
    fname = FILES[which]
    src = hf_hub_download(repo_id=REPO_ID, filename=f"{SUBFOLDER}/{fname}",
                          local_dir=OUT_DIR, local_dir_use_symlinks=False)
    dst = os.path.join(OUT_DIR, fname)
    if os.path.abspath(src) != os.path.abspath(dst):
        shutil.copy(src, dst)
    return dst


def download_direct(which):
    import urllib.request
    fname = FILES[which]
    url = f"https://huggingface.co/{REPO_ID}/resolve/main/{SUBFOLDER}/{fname}"
    dst = os.path.join(OUT_DIR, fname)
    print(f"  downloading: {url}")
    urllib.request.urlretrieve(url, dst)
    return dst


def verify(path):
    try:
        import torch
    except ImportError:
        print("  (torch not available, skipping verify)")
        return
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ck, dict):
        print(f"  keys: {list(ck.keys())[:5]}")
        sd = ck.get("model", ck.get("state_dict", ck))
        img = [k for k in sd if "image_encoder" in k]
        print(f"  tensors: {len(sd)}, image_encoder: {len(img)}")
        if img:
            print(f"  sample: {img[0]}")
        else:
            print("  warning: no 'image_encoder' keys found")


def list_repo():
    from huggingface_hub import list_repo_files
    print(f"files in {REPO_ID}:")
    for f in list_repo_files(REPO_ID):
        print(f"  {f}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    args = ["b5"]
    if "list" in args:
        list_repo(); return
    which = list(FILES) if "both" in args else [a for a in args if a in FILES] or ["b5"]
    for w in which:
        print(f"downloading {w} from {REPO_ID}")
        try:
            try:
                path = download_hub(w)
            except ImportError:
                path = download_direct(w)
            print(f"  saved: {path}")
            verify(path)
        except Exception as e:
            print(f"  error: {e}")



if __name__ == "__main__":
    main()