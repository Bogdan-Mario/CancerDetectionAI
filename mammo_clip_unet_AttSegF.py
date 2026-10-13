"""
Mammo-CLIP EfficientNet (efficientnet_pytorch) encoder in a U-Net / Attention-U-Net /
SegFormer-style-MLP decoder for microcalcification segmentation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

BACKBONES = {"b5": "efficientnet-b5", "b2": "efficientnet-b2"}
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD  = (0.229, 0.224, 0.225)


class _AttnGate(nn.Module):
    def __init__(self, g_ch, x_ch, inter):
        super().__init__()
        self.wg = nn.Conv2d(g_ch, inter, 1); self.wx = nn.Conv2d(x_ch, inter, 1)
        self.psi = nn.Conv2d(inter, 1, 1); self.relu = nn.ReLU(True); self.sig = nn.Sigmoid()
    def forward(self, g, x):
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return x * self.sig(self.psi(self.relu(self.wg(g) + self.wx(x))))


class _Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, attention=False):
        super().__init__()
        self.attn = _AttnGate(in_ch, skip_ch, max(skip_ch // 2, 1)) if (attention and skip_ch > 0) else None
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, 1, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(True))
    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            if self.attn is not None:
                skip = self.attn(x, skip)
            x = torch.cat([x, skip], 1)
        return self.conv(x)


class _MLPHead(nn.Module):        # SegFormer-style all-MLP decode head
    def __init__(self, chs, embed=256, out_channels=1):
        super().__init__()
        self.proj = nn.ModuleList([nn.Conv2d(c, embed, 1) for c in chs])
        self.fuse = nn.Sequential(nn.Conv2d(embed * len(chs), embed, 1, bias=False),
                                  nn.BatchNorm2d(embed), nn.ReLU(True))
        self.head = nn.Conv2d(embed, out_channels, 1)
    def forward(self, feats):
        size = feats[0].shape[-2:]
        ups = [F.interpolate(p(f), size=size, mode="bilinear", align_corners=False)
               for p, f in zip(self.proj, feats)]
        return self.head(self.fuse(torch.cat(ups, 1)))


class MammoClipUNet(nn.Module):
    def __init__(self, backbone="efficientnet-b5", out_channels=1,
                 decoder="unet", decoder_channels=(256, 128, 64, 32, 16), mlp_embed=256):
        super().__init__()
        self.decoder_type = decoder
        self.encoder = EfficientNet.from_name(backbone, in_channels=3, image_size=None)
        self._keys = [f"reduction_{i}" for i in range(1, 6)]
        with torch.no_grad():
            eps = self.encoder.extract_endpoints(torch.zeros(1, 3, 128, 128))
        chs = [eps[k].shape[1] for k in self._keys]
        if decoder in ("unet", "attention"):
            self.blocks = nn.ModuleList()
            in_ch = chs[-1]; skip_chs = chs[-2::-1] + [0]
            for i, out_ch in enumerate(decoder_channels):
                self.blocks.append(_Up(in_ch, skip_chs[i], out_ch, attention=(decoder == "attention")))
                in_ch = out_ch
            self.head = nn.Conv2d(in_ch, out_channels, 1)
        elif decoder == "mlp":
            self.head = _MLPHead(chs, embed=mlp_embed, out_channels=out_channels)
        else:
            raise ValueError(f"unknown decoder: {decoder}")

    def forward(self, inp):
        eps = self.encoder.extract_endpoints(inp)
        feats = [eps[k] for k in self._keys]
        if self.decoder_type in ("unet", "attention"):
            skips = feats[-2::-1]; x = feats[-1]
            for i, blk in enumerate(self.blocks):
                x = blk(x, skips[i] if i < len(skips) else None)
            out = self.head(x)
        else:
            out = self.head(feats)
        if out.shape[-2:] != inp.shape[-2:]:
            out = F.interpolate(out, size=inp.shape[-2:], mode="bilinear", align_corners=False)
        return out


def load_mammo_clip_encoder(model, ckpt_path, verbose=True):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck.get("state_dict", ck))
    target = model.encoder.state_dict(); tgt = set(target.keys())
    enc = {}
    for k, v in sd.items():
        if "image_encoder." not in k:
            continue
        nk = k.split("image_encoder.", 1)[1]
        if nk in tgt and target[nk].shape == v.shape:
            enc[nk] = v
    model.encoder.load_state_dict(enc, strict=False)
    if verbose:
        print(f"[Mammo-CLIP] matched {len(enc)}/{len(tgt)} encoder tensors")
    if len(enc) == 0:
        print("  sample checkpoint keys:", [k for k in sd if 'image_encoder' in k][:6])
        print("  sample target keys   :", list(tgt)[:6])
        raise RuntimeError("No Mammo-CLIP encoder weights matched.")
    return model


def load_imagenet_encoder(model, backbone):
    ref = EfficientNet.from_pretrained(backbone, in_channels=3)
    src, target = ref.state_dict(), model.encoder.state_dict()
    enc = {k: v for k, v in src.items() if k in target and target[k].shape == v.shape}
    model.encoder.load_state_dict(enc, strict=False)
    print(f"[ImageNet] matched {len(enc)}/{len(target)} encoder tensors")
    return model


class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        p = torch.sigmoid(logits).view(-1); t = targets.view(-1)
        inter = (p * t).sum()
        dice = 1 - (2 * inter + 1) / (p.sum() + t.sum() + 1)
        return bce + dice


def dice_iou(gt, pred):
    inter = float((gt * pred).sum()); gs, ps = float(gt.sum()), float(pred.sum())
    d = (2 * inter) / (gs + ps) if (gs + ps) > 0 else 1.0
    i = inter / (gs + ps - inter) if (gs + ps - inter) > 0 else 1.0
    return d, i