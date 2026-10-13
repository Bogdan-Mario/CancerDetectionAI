"""
VLM core wrapper (BiomedCLIP) — the shared semantic engine for all three stages.

BiomedCLIP (microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) is a
CLIP-style vision-language model: a ViT-B/16 image encoder + a PubMedBERT text
encoder projected into a shared embedding space. It is the same model already
referenced in PatchFilterClassificationTrainingAndTesting.get_model("medclip"),
so the pipeline stays self-consistent.

This module exposes the three primitives the assignment asks for:
  * image / text EMBEDDINGS in a shared space           -> encode_image / encode_text
  * image-text SIMILARITY scores (zero-shot logits)     -> similarity / classify_zeroshot
  * RELEVANCE / heat maps for a text query              -> relevance_map

Everything downstream (classification fusion, detection reranking, segmentation
prompting) is built on top of these primitives.

Requires:  pip install open_clip_torch torch pillow numpy
"""
import numpy as np
import torch
import torch.nn.functional as F

_MODEL_TAG = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


def _to_pil_rgb(patch):
    """numpy grayscale/RGB uint8 (or float [0,1]) -> PIL RGB."""
    from PIL import Image
    a = patch
    if a.dtype != np.uint8:
        a = np.clip(a * 255.0 if a.max() <= 1.0 else a, 0, 255).astype(np.uint8)
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    return Image.fromarray(a)


class VLM:
    """Thin BiomedCLIP wrapper with mammography-friendly helpers."""

    def __init__(self, device=None, model_tag=_MODEL_TAG):
        import open_clip
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_tag)
        self.tokenizer = open_clip.get_tokenizer(model_tag)
        self.model = self.model.to(self.device).eval()
        # CLIP logit scale (temperature). exp() gives the multiplier used on cosine sims.
        self.logit_scale = self.model.logit_scale.exp().item()

    # ---- embeddings -------------------------------------------------------
    @torch.no_grad()
    def encode_image(self, patches):
        """patches: single numpy array or list of numpy arrays -> (N, D) L2-normalized."""
        if isinstance(patches, np.ndarray) and patches.ndim in (2, 3) and not isinstance(patches, list):
            patches = [patches]
        batch = torch.stack([self.preprocess(_to_pil_rgb(p)) for p in patches]).to(self.device)
        feats = self.model.encode_image(batch)
        return F.normalize(feats, dim=-1)

    @torch.no_grad()
    def encode_text(self, prompts):
        """prompts: list[str] -> (N, D) L2-normalized text embeddings."""
        toks = self.tokenizer(prompts).to(self.device)
        feats = self.model.encode_text(toks)
        return F.normalize(feats, dim=-1)

    @torch.no_grad()
    def build_text_bank(self, prompt_dict):
        """
        prompt_dict: {key: [prompt, ...]} -> (keys, bank) where bank is (K, D)
        L2-normalized, one averaged prototype per key (prompt ensembling).
        """
        keys, protos = [], []
        for key, prompts in prompt_dict.items():
            emb = self.encode_text(list(prompts))          # (P, D)
            protos.append(F.normalize(emb.mean(dim=0), dim=-1))
            keys.append(key)
        return keys, torch.stack(protos)                   # (K, D)

    # ---- similarity / zero-shot ------------------------------------------
    @torch.no_grad()
    def similarity(self, image_emb, text_bank, softmax=True):
        """
        image_emb: (N, D), text_bank: (K, D) -> (N, K) scaled cosine logits.
        With softmax=True returns a probability distribution over the K prototypes.
        """
        logits = self.logit_scale * image_emb @ text_bank.t()
        return F.softmax(logits, dim=-1) if softmax else logits

    @torch.no_grad()
    def classify_zeroshot(self, patches, prompt_dict):
        """
        End-to-end zero-shot: patches + class prompt dict -> (probs, keys).
        probs: (N, K) numpy, keys: list of class ids in column order.
        """
        keys, bank = self.build_text_bank(prompt_dict)
        img = self.encode_image(patches)
        probs = self.similarity(img, bank, softmax=True)
        return probs.cpu().numpy(), keys

    # ---- relevance / heat map --------------------------------------------
    @torch.no_grad()
    def relevance_map(self, patch, prompts, win=64, stride=32, pos_neg=None):
        """
        Sliding-window relevance map: how strongly each region matches `prompts`.

        Model-agnostic and robust — we crop windows, embed each, and score its
        similarity to the (ensembled) positive text prototype. If `pos_neg` is a
        (pos_prompts, neg_prompts) tuple, the score is a softmax margin
        pos-vs-neg (better calibrated for detection/segmentation guidance).

        Returns a float32 heatmap the same H,W as `patch`, values in [0,1].
        """
        H, W = patch.shape[:2]
        if pos_neg is not None:
            pos_p, neg_p = pos_neg
            pos = self.encode_text(list(pos_p)).mean(0, keepdim=True)
            neg = self.encode_text(list(neg_p)).mean(0, keepdim=True)
            bank = F.normalize(torch.cat([pos, neg], 0), dim=-1)     # (2, D)
        else:
            pos = F.normalize(self.encode_text(list(prompts)).mean(0, keepdim=True), dim=-1)
            bank = pos                                               # (1, D)

        heat = np.zeros((H, W), np.float32)
        cnt = np.zeros((H, W), np.float32)
        coords, crops = [], []
        for y in range(0, max(1, H - win + 1), stride):
            for x in range(0, max(1, W - win + 1), stride):
                crops.append(patch[y:y + win, x:x + win])
                coords.append((x, y))
        if not crops:
            return heat
        emb = self.encode_image(crops)                              # (M, D)
        logits = self.logit_scale * emb @ bank.t()                  # (M, 1 or 2)
        if bank.shape[0] == 2:
            score = F.softmax(logits, dim=-1)[:, 0]                 # pos probability
        else:
            score = torch.sigmoid(logits[:, 0] - logits.mean())     # relative activation
        score = score.cpu().numpy()
        for (x, y), s in zip(coords, score):
            heat[y:y + win, x:x + win] += s
            cnt[y:y + win, x:x + win] += 1.0
        cnt[cnt == 0] = 1.0
        heat /= cnt
        m, M = heat.min(), heat.max()
        return (heat - m) / (M - m + 1e-6)


def load_vlm(device=None):
    """Convenience singleton-style loader."""
    return VLM(device=device)
