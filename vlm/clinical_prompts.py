"""
Clinical text descriptions used as the language side of the VLM.

The classifier (PatchFilterClassificationTrainingAndTesting.py, balanced mode)
produces 4 classes on each square patch:
    0 = healthy / normal tissue
    1 = mass
    2 = calcification (suspicious microcalcifications)
    3 = both (mass + calcification)

For robust zero-shot similarity we do *prompt ensembling*: every class is
described by several short radiology-style sentences, their text embeddings are
averaged into a single normalized class prototype. This is the standard CLIP
recipe and is markedly more stable than a single hand-written sentence.

`EXTRA_PROMPTS` holds auxiliary concepts (artifact, dense tissue) mentioned in
the assignment. They are not classifier targets but are useful as:
  * an "artifact" reject prototype during detection false-positive cleanup,
  * a "dense tissue" context prototype for difficulty analysis.
"""

# One entry per classifier label (index == class id). Order matters.
CLASS_PROMPTS = {
    0: [  # healthy / normal
        "a mammogram patch of normal breast tissue",
        "normal fibroglandular breast tissue, no lesion",
        "a benign mammographic region without abnormality",
        "healthy breast parenchyma on mammography",
    ],
    1: [  # mass
        "a mammogram patch showing a breast mass",
        "a soft-tissue mass lesion on mammography",
        "a well-defined or spiculated breast mass",
        "a suspicious mass in dense breast tissue",
    ],
    2: [  # calcification / microcalcifications
        "a mammogram patch with suspicious microcalcifications",
        "clustered pleomorphic microcalcifications on mammography",
        "fine granular calcifications in breast tissue",
        "a region containing suspicious calcifications",
    ],
    3: [  # both
        "a breast mass with associated microcalcifications",
        "a mammographic mass containing clustered calcifications",
        "a suspicious lesion showing both a mass and calcifications",
    ],
}

# Human-readable class names (align with the classifier).
CLASS_NAMES = {0: "healthy", 1: "mass", 2: "calcification", 3: "both"}

# Auxiliary concepts (not classifier targets).
EXTRA_PROMPTS = {
    "artifact": [
        "an imaging artifact on a mammogram",
        "a skin marker or scanning artifact, not a lesion",
        "a metal clip or label overlay on mammography",
    ],
    "dense_tissue": [
        "extremely dense breast tissue on mammography",
        "heterogeneously dense fibroglandular tissue",
    ],
}

# Binary prompt bank for the DETECTION stage: "is this crop a real
# microcalcification?" — used to validate / rerank RetinaNet boxes.
DETECTION_POS_PROMPTS = [
    "a crop containing real suspicious microcalcifications",
    "clustered microcalcifications on mammography",
    "fine pleomorphic calcifications inside the breast",
]
DETECTION_NEG_PROMPTS = [
    "normal breast tissue without calcifications",
    "an imaging artifact, not a calcification",
    "empty background outside the breast",
]

# Text prompt(s) used to build the SEGMENTATION relevance map.
SEGMENTATION_PROMPTS = DETECTION_POS_PROMPTS


def num_classes():
    return len(CLASS_PROMPTS)


def sick_only_prompts():
    """Prompts for 'sick_only' balance mode (3 classes: mass, calc, both)."""
    return {0: CLASS_PROMPTS[1], 1: CLASS_PROMPTS[2], 2: CLASS_PROMPTS[3]}
