"""
score_fusion.py

Combines the four module scores (PRNU, ELA, Freq, Metadata) into a final verdict.
Supports two image types: "general" and "face", each with different weights
reflecting model performance on that image category.
"""

from __future__ import annotations

# Module weights per image type

WEIGHTS_GENERAL = {
    "prnu":     0.45,
    "ela":      0.25,
    "freq":     0.25,
    "metadata": 0.05,
}

WEIGHTS_FACE = {
    "prnu":     0.35,
    "ela":      0.20,
    "freq":     0.35,
    "metadata": 0.10,
}

# Score at which the verdict switches from Real to AI-Generated
DECISION_THRESHOLD = 0.50

# Fraction of metadata weight to redistribute when PNG format reduces its reliability
PNG_METADATA_WEIGHT_REDUCTION = 0.5


def fuse_scores(
    prnu_score:      float | None,
    ela_score:       float | None,
    freq_score:      float | None,
    metadata_score:  float,
    metadata_format: str = "JPEG",
    metadata_reason: str = "",
    image_type:      str = "general",
) -> dict:
    """
    Combines all module scores into a final verdict.

    If any CNN module failed (score is None), its weight is redistributed
    proportionally across the remaining active modules.
    """
    scores = {
        "prnu":     prnu_score,
        "ela":      ela_score,
        "freq":     freq_score,
        "metadata": metadata_score,
    }

    # Select the weight set for this image type
    weights = dict(WEIGHTS_FACE if image_type == "face" else WEIGHTS_GENERAL)

    # Reduce metadata weight for PNG images unless a strong AI signal is present.
    # PNG does not embed EXIF the same way as JPEG, making the metadata check unreliable.
    if (metadata_format == "PNG"
            and "ai tool" not in metadata_reason.lower()
            and "suspicious" not in metadata_reason.lower()):
        freed = weights["metadata"] * PNG_METADATA_WEIGHT_REDUCTION
        weights["metadata"] -= freed
        for k in ("prnu", "ela", "freq"):
            weights[k] += freed / 3.0

    # Redistribute weight from failed CNN modules to the remaining active ones
    missing      = [k for k in ("prnu", "ela", "freq") if scores[k] is None]
    active       = [k for k in scores if scores[k] is not None]

    if not active:
        return {
            "final_score":   0.5,
            "verdict":       "Uncertain",
            "confidence":    0.0,
            "weights_used":  weights,
            "module_scores": scores,
        }

    if missing:
        freed_weight = sum(weights[k] for k in missing)
        active_total = sum(weights[k] for k in active)
        for k in active:
            weights[k] += freed_weight * (weights[k] / active_total)
        for k in missing:
            weights[k] = 0.0

    # Normalize so weights always sum to 1.0
    total   = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}

    # Weighted average of all active scores
    final_score = sum(
        scores[k] * weights[k]
        for k in scores
        if scores[k] is not None
    )
    final_score = round(float(final_score), 4)

    verdict    = "AI-Generated" if final_score >= DECISION_THRESHOLD else "Real"
    distance   = abs(final_score - DECISION_THRESHOLD)
    confidence = round(50.0 + distance * 100.0, 1)

    return {
        "final_score":   final_score,
        "verdict":       verdict,
        "confidence":    confidence,
        "weights_used":  {k: round(v, 4) for k, v in weights.items()},
        "module_scores": scores,
        "image_type":    image_type,
    }
