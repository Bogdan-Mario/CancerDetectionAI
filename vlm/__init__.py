"""VLM integration package (Tema 2): BiomedCLIP semantic layer over the
classification / detection / segmentation pipeline.

Note: importing this package is kept lightweight (no torch). Import the VLM
wrapper explicitly where needed:  `from vlm.vlm_core import VLM`.
The framework-free ops in `vlm.fusion_ops` import with numpy only.
"""

__all__ = ["VLM", "load_vlm"]


def __getattr__(name):        # lazy: only pull in torch when VLM is actually used
    if name in ("VLM", "load_vlm"):
        from vlm.vlm_core import VLM, load_vlm
        return {"VLM": VLM, "load_vlm": load_vlm}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
