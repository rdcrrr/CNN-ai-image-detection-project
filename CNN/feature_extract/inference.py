"""
inference.py
------------
Loads the correct set of CNN models based on image_type ("general" or "face")
and runs all four modules on a single image.

Place this file at:  imageMetadata/CNN/feature_extract/inference.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Path setup
# CNN/feature_extract/ -> CNN/ -> imageMetadata/ (project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CNN.feature_extract.PRNU_and_ELA_preparing import build_prnu_tensor, build_ela_tensor
from CNN.feature_extract.frequency_extractor import build_frequency_tensor
from metadata_extract.metadata_scorer import analyze_metadata

# ---------------------------------------------------------------------------
# 1.  CNN architecture definitions  (must match cnn_training.py exactly)
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class _BaseCNN(nn.Module):
    def __init__(self, in_channels: int, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32),
            ConvBlock(32,  64),
            ConvBlock(64,  128),
            ConvBlock(128, 256),
            ConvBlock(256, 256),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x)).squeeze(1)


class PRNUNet(_BaseCNN):
    def __init__(self): super().__init__(in_channels=2)

class ELANet(_BaseCNN):
    def __init__(self): super().__init__(in_channels=1)

class FreqNet(_BaseCNN):
    def __init__(self): super().__init__(in_channels=3)


# ---------------------------------------------------------------------------
# 2.  Model paths — one set per image type
# ---------------------------------------------------------------------------
SAVED_DIR = PROJECT_ROOT / "CNN" / "saved_modules"

MODEL_PATHS = {
    "general": {
        "prnu": SAVED_DIR / "prnu_general_v2.pth",
        "ela":  SAVED_DIR / "ela_general_v2.pth",
        "freq": SAVED_DIR / "freq_general_v2.pth",
    },
    "face": {
        "prnu": SAVED_DIR / "prnu_faces_v2.pth",
        "ela":  SAVED_DIR / "ela_faces_v2.pth",
        "freq": SAVED_DIR / "freq_faces_v2.pth",
    },
}

TARGET_SIZE = (224, 224)

# ---------------------------------------------------------------------------
# 3.  Model loader  (singleton per name+type — load once, reuse)
# ---------------------------------------------------------------------------
_loaded_models: dict = {}


def _load_model(key: str, model_class, path: Path, device):
    if key not in _loaded_models:
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        model = model_class()
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        _loaded_models[key] = model
        print(f"[Inference] Loaded {key} from {path.name}")
    return _loaded_models[key]


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_all_models():
    """Pre-loads all six models. Call once at server startup."""
    device = get_device()
    for image_type, paths in MODEL_PATHS.items():
        _load_model(f"prnu_{image_type}", PRNUNet, paths["prnu"], device)
        _load_model(f"ela_{image_type}",  ELANet,  paths["ela"],  device)
        _load_model(f"freq_{image_type}", FreqNet, paths["freq"], device)
    print("[Inference] All models loaded.")


# ---------------------------------------------------------------------------
# 4.  Inference helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def _run_cnn(model, tensor, device) -> float:
    inp   = tensor.unsqueeze(0).to(device)
    logit = model(inp)
    return float(torch.sigmoid(logit).item())


def _safe_cnn_score(key, model_class, path, tensor_fn,
                    image_path, device, is_freq=False) -> float | None:
    try:
        if is_freq:
            tensor, _ = tensor_fn(image_path, TARGET_SIZE)
        else:
            tensor = tensor_fn(image_path, TARGET_SIZE)
        model = _load_model(key, model_class, path, device)
        return _run_cnn(model, tensor, device)
    except Exception as e:
        print(f"[Inference] Warning — {key} failed: {e}")
        return None


# ---------------------------------------------------------------------------
# 5.  Public function
# ---------------------------------------------------------------------------
def run_inference(image_path: str, image_type: str = "general") -> dict:
    """
    Runs all four modules on a single image.

    Parameters:
        image_path : path to the image file
        image_type : "general" or "face"

    Returns dict with prnu_score, ela_score, freq_score,
    metadata_score, metadata_reason, metadata_format.
    """
    if image_type not in MODEL_PATHS:
        raise ValueError(f"image_type must be 'general' or 'face', got '{image_type}'")

    device = get_device()
    paths  = MODEL_PATHS[image_type]

    prnu_score = _safe_cnn_score(
        f"prnu_{image_type}", PRNUNet, paths["prnu"],
        build_prnu_tensor, image_path, device
    )
    ela_score = _safe_cnn_score(
        f"ela_{image_type}", ELANet, paths["ela"],
        build_ela_tensor, image_path, device
    )
    freq_score = _safe_cnn_score(
        f"freq_{image_type}", FreqNet, paths["freq"],
        build_frequency_tensor, image_path, device, is_freq=True
    )

    meta = analyze_metadata(image_path)

    return {
        "prnu_score":      prnu_score,
        "ela_score":       ela_score,
        "freq_score":      freq_score,
        "metadata_score":  meta["score"],
        "metadata_reason": meta["reason"],
        "metadata_format": meta.get("format", "UNKNOWN"),
        "image_type":      image_type,
    }