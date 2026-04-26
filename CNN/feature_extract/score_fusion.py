"""
score_fusion

Combines the four module scores into a final verdict.
Supports two image types: "general" and "face", each with
different weights reflecting model performance on that type.


"""

from __future__ import annotations


# Two different weights for the two module groups
WEIGHTS_GENERAL = {
    "prnu":     0.45,   # strongest on general images
    "ela":      0.10,   # weakest on general images
    "freq":     0.35,   # strong on general images
    "metadata": 0.10,
}

WEIGHTS_FACE = {
    "prnu":     0.35,
    "ela":      0.20,
    "freq":     0.35,
    "metadata": 0.10,
}

THRESHOLD = 0.50



def fuse_scores(
    prnu_score:      float | None,
    ela_score:       float | None,
    freq_score:      float | None,
    metadata_score:  float,
    metadata_format: str = "JPEG",
    metadata_reason: str = "",
    image_type:      str = "general",   # "general" or "face"
) -> dict:
    """
    Combines all module scores into a final verdict.
    """
    scores = {
        "prnu":     prnu_score,
        "ela":      ela_score,
        "freq":     freq_score,
        "metadata": metadata_score,
    }

    # Select correct weight set based on image type
    if image_type == "face":
        weights = dict(WEIGHTS_FACE)
    else:
        weights = dict(WEIGHTS_GENERAL)

    # Reduce metadata weight for PNG
    if (metadata_format == "PNG"
            and "ai tool" not in metadata_reason.lower()
            and "suspicious" not in metadata_reason.lower()):
        freed = weights["metadata"] * 0.5
        weights["metadata"] -= freed
        for k in ("prnu", "ela", "freq"):
            weights[k] += freed / 3.0

    # Redistribute weight from any failed CNN modules
    missing = [k for k in ("prnu", "ela", "freq") if scores[k] is None]
    active  = [k for k in scores if scores[k] is not None]

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

    # Normalise weights to sum to 1.0
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}

    # Weighted average
    final_score = sum(
        scores[k] * weights[k]
        for k in scores
        if scores[k] is not None
    )
    final_score = round(float(final_score), 4)

    verdict    = "AI-Generated" if final_score >= THRESHOLD else "Real"
    distance   = abs(final_score - THRESHOLD)
    confidence = round(50.0 + distance * 100.0, 1)

    return {
        "final_score":   final_score,
        "verdict":       verdict,
        "confidence":    confidence,
        "weights_used":  {k: round(v, 4) for k, v in weights.items()},
        "module_scores": scores,
        "image_type":    image_type,
    }