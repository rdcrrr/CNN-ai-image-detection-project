"""
project_graphs.py
-----------------
Generates clean, simple presentation graphs for the AI Image Detection project.
White background, clear labels, no styling — Google Colab notebook style.

Run with:
  python project_graphs.py

Saves all graphs as PNG files in a 'graphs/' folder.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ── Output folder ──────────────────────────────────────────────────────────
OUTPUT_DIR = Path("graphs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Reset to default matplotlib style (white background, simple) ───────────
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"]   = "white"
plt.rcParams["font.size"]        = 11

# ── Data ───────────────────────────────────────────────────────────────────

ACCURACY = {
    "general": {"PRNU": 93.4, "ELA": 76.1, "FREQ": 87.7},
    "face":    {"PRNU": 91.6, "ELA": 88.2, "FREQ": 78.9},
}

AUC = {
    "general": {"PRNU": 0.9826, "ELA": 0.8269, "FREQ": 0.9575},
    "face":    {"PRNU": 0.9732, "ELA": 0.9520, "FREQ": 0.8687},
}

WEIGHTS_GENERAL = {"PRNU": 0.45, "ELA": 0.10, "FREQ": 0.35, "Metadata": 0.10}
WEIGHTS_FACE    = {"PRNU": 0.35, "ELA": 0.20, "FREQ": 0.35, "Metadata": 0.10}

# Real scores from actual test runs
EXAMPLE_AI_SCORES = {
    "PRNU":     0.9996,
    "ELA":      0.9755,
    "FREQ":     0.9875,
    "Metadata": 0.35,
}

EXAMPLE_REAL_SCORES = {
    "PRNU":     0.0036,
    "ELA":      0.9925,
    "FREQ":     0.2270,
    "Metadata": 0.35,
}


# ── Graph 1: Accuracy comparison ───────────────────────────────────────────
def plot_accuracy():
    fig, ax = plt.subplots(figsize=(8, 5))

    models  = ["PRNU", "ELA", "FREQ"]
    x       = np.arange(len(models))
    width   = 0.35

    bars_g = ax.bar(x - width/2, [ACCURACY["general"][m] for m in models],
                    width, label="General Mode", color="steelblue")
    bars_f = ax.bar(x + width/2, [ACCURACY["face"][m] for m in models],
                    width, label="Face Mode", color="darkorange")

    # Value labels on top of bars
    for bar in bars_g:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=10)
    for bar in bars_f:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=10)

    ax.set_title("Test Accuracy by Module and Mode")
    ax.set_xlabel("CNN Module")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(["PRNU\n(Noise Residual)", "ELA\n(Error Level Analysis)", "FREQ\n(Frequency Analysis)"])
    ax.set_ylim(60, 102)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    path = OUTPUT_DIR / "1_accuracy_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── Graph 2: AUC comparison ────────────────────────────────────────────────
def plot_auc():
    fig, ax = plt.subplots(figsize=(8, 5))

    models = ["PRNU", "ELA", "FREQ"]
    x      = np.arange(len(models))
    width  = 0.35

    bars_g = ax.bar(x - width/2, [AUC["general"][m] for m in models],
                    width, label="General Mode", color="steelblue")
    bars_f = ax.bar(x + width/2, [AUC["face"][m] for m in models],
                    width, label="Face Mode", color="darkorange")

    for bar in bars_g:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10)
    for bar in bars_f:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10)

    ax.set_title("Test AUC by Module and Mode\n(AUC = Area Under ROC Curve, higher is better)")
    ax.set_xlabel("CNN Module")
    ax.set_ylabel("AUC Score (0 = random, 1 = perfect)")
    ax.set_xticks(x)
    ax.set_xticklabels(["PRNU\n(Noise Residual)", "ELA\n(Error Level Analysis)", "FREQ\n(Frequency Analysis)"])
    ax.set_ylim(0.75, 1.02)
    ax.axhline(y=0.90, color="gray", linestyle="--", alpha=0.6, label="0.90 reference line")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    path = OUTPUT_DIR / "2_auc_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── Graph 3: Score fusion weights ──────────────────────────────────────────
def plot_weights():
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    colors = ["steelblue", "darkorange", "green", "gray"]

    for ax, (title, weights) in zip(axes, [
        ("General Mode", WEIGHTS_GENERAL),
        ("Face Mode",    WEIGHTS_FACE),
    ]):
        labels = list(weights.keys())
        values = list(weights.values())

        bars = ax.bar(labels, values, color=colors, width=0.5)

        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{bar.get_height():.0%}", ha="center", va="bottom", fontsize=11)

        ax.set_title(f"Score Fusion Weights\n{title}")
        ax.set_xlabel("Analysis Module")
        ax.set_ylabel("Weight (contribution to final score)")
        ax.set_ylim(0, 0.65)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    path = OUTPUT_DIR / "3_score_fusion_weights.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── Graph 4: Module scores on example images ───────────────────────────────
def plot_example_scores():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    modules = ["PRNU", "ELA", "FREQ", "Metadata"]

    for ax, (title, scores, verdict) in zip(axes, [
        ("Known AI-Generated Image\n(fake_100.jpg — Face Mode)",
         EXAMPLE_AI_SCORES, "Verdict: AI-Generated | Confidence: 89.4%"),
        ("Known Real Image\n(real_57.jpg — Face Mode)",
         EXAMPLE_REAL_SCORES, "Verdict: Real | Confidence: 68.6%"),
    ]):
        values = [scores[m] for m in modules]

        # Color each bar based on whether it's above or below threshold
        bar_colors = ["tomato" if v >= 0.5 else "mediumseagreen" for v in values]
        bars = ax.bar(modules, values, color=bar_colors, width=0.5, edgecolor="black", linewidth=0.5)

        # Value labels with more decimal places to avoid showing 0.00
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=10)

        ax.axhline(y=0.5, color="black", linestyle="--", linewidth=1.2,
                   label="Decision threshold (0.5)")
        ax.set_title(title)
        ax.set_xlabel("Analysis Module")
        ax.set_ylabel("AI Probability Score\n(0 = Real, 1 = AI-Generated)")
        ax.set_ylim(0, 1.2)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend(fontsize=9)

        # Verdict text below title
        ax.text(0.5, -0.18, verdict,
                transform=ax.transAxes,
                ha="center", va="center", fontsize=10,
                style="italic")

        # Legend for bar colors
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="tomato",        label="Above threshold → AI signal"),
            Patch(facecolor="mediumseagreen",label="Below threshold → Real signal"),
        ]
        ax.legend(handles=legend_elements, fontsize=8, loc="upper right")

    plt.tight_layout()
    path = OUTPUT_DIR / "4_example_scores.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── Graph 5: Training dataset composition ─────────────────────────────────
def plot_dataset():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # General dataset sources
    general_sources = ["CIFAKE\n(Stable Diffusion)", "DALL-E", "Midjourney",
                       "Stable Diffusion\n(diverse)", "NeuralTextures"]
    general_counts  = [10000, 2000, 930, 2098, 1000]

    # Face dataset sources
    face_sources = ["StyleGAN\nFaces", "DeepFaceLab", "Face2Face",
                    "FaceShifter", "FaceSwap"]
    face_counts  = [1000, 1606, 1000, 1000, 1600]

    for ax, (title, sources, counts, color) in zip(axes, [
        ("General AI Training Data\n(AI images by source)", general_sources, general_counts, "steelblue"),
        ("Face AI Training Data\n(AI images by source)",    face_sources,    face_counts,    "darkorange"),
    ]):
        bars = ax.barh(sources, counts, color=color, edgecolor="black", linewidth=0.5)

        for bar, count in zip(bars, counts):
            ax.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2,
                    f"{count:,}", va="center", fontsize=10)

        ax.set_title(title)
        ax.set_xlabel("Number of Images")
        ax.set_ylabel("Image Source / Generator")
        ax.set_xlim(0, max(counts) * 1.25)
        ax.grid(axis="x", linestyle="--", alpha=0.5)

    plt.tight_layout()
    path = OUTPUT_DIR / "5_dataset_composition.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── Run all ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating graphs...\n")
    plot_accuracy()
    plot_auc()
    plot_weights()
    plot_example_scores()
    plot_dataset()
    print(f"\nAll graphs saved to: {OUTPUT_DIR.resolve()}")