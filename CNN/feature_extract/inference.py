"""
inference.py

Loads the correct set of CNN models based on image_type (general,face)
and runs all four modules on a single image.

Image uploaded
      |
      v
app.py -- validate, save to temp file
      |
      v
      |-----> PRNU pipeline
      |       load -> grayscale -> wavelet -> denoise ->
      |       reconstruct -> subtract -> clean -> normalize ->
      |       stack (2,224,224) -> PRNUNet -> sigmoid -> 0.82
      |
      |-----> ELA pipeline
      |       load -> recompress -> subtract -> normalize ->
      |       (1,224,224) -> ELANet -> sigmoid -> 0.71
      |
      |-----> Frequency pipeline
      |       load -> grayscale -> FFT -> DCT -> azimuthal ->
      |       normalize -> stack (3,224,224) -> FreqNet -> sigmoid -> 0.68
      |
      |-----> Metadata pipeline
              raw bytes -> find EXIF -> find software tag ->
              check AI keywords -> 0.05
      |
      v
score_fusion -- weighted average -> 0.719
      |
      v
verdict: AI-Generated, confidence: 71.9%
      |
      v
JSON response to user, temp file deleted

"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

# Path setup so Python can find the other project files
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import the three feature extractors we already covered
from CNN.feature_extract.PRNU_and_ELA_preparing import build_prnu_tensor, build_ela_tensor
from CNN.feature_extract.frequency_extractor    import build_frequency_tensor
from metadata_extract.metadata_scorer           import analyze_metadata



# CNN ARCHITECTURE


# One reusable building block used by all three CNNs
# in_ch, how many channels come in
# out_ch, how many channels come out
# pool, whether to halve the spatial dimensions after
class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, pool: bool = True):
        super().__init__()
        layers = [
            # Slides a 3x3 filter across the input looking for patterns
            # padding=1 keeps the output the same size as the input
            # bias=False because BatchNorm handles the offset instead
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            # Normalizes the values coming out of the conv layer
            # keeps numbers in a stable range so training does not explode
            nn.BatchNorm2d(out_ch),
            # Activation function, replaces negative values with zero
            # this lets the network learn non linear patterns
            nn.ReLU(inplace=True),
        ]
        if pool:
            # Takes the maximum value in each 2x2 region
            # halves the image size while keeping the strongest signals
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# Shared base class for all three CNNs
# the only difference between PRNUNet, ELANet, FreqNet is how many
# input channels they accept, so we put all the shared code here
class _BaseCNN(nn.Module):
    def __init__(self, in_channels: int, dropout: float = 0.3):
        super().__init__()
        """
        Pytourch does not save the architecture of the CNN only the weight.
        requiring to build a replica to properly load in the models
        
        """

        # Five conv blocks that progressively shrink the image
        # and increase the number of feature maps
        # spatial size goes: 224 -> 112 -> 56 -> 28 -> 14 -> 7
        # feature maps go:   in  ->  32 -> 64 -> 128 -> 256 -> 256
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32),
            ConvBlock(32,  64),
            ConvBlock(64,  128),
            ConvBlock(128, 256),
            ConvBlock(256, 256),
        )

        # After the conv blocks the output is 256 feature maps each 7x7
        # we flatten that into a single long vector and run it through
        # fully connected layers to produce one final number
        self.classifier = nn.Sequential(
            # Turns the (256, 7, 7) tensor into a flat vector of length 256*7*7 = 12544
            nn.Flatten(),
            # First fully connected layer, compresses 12544 down to 512
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            # Randomly zeroes out some neurons during training
            # forces the network not to rely on any single neuron
            # disabled automatically during eval mode
            nn.Dropout(dropout),
            # Second fully connected layer, compresses 512 down to 128
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            # Final layer outputs one single number (the raw logit)
            # positive = leans toward AI, negative = leans toward real
            nn.Linear(128, 1),
        )

    def forward(self, x):
        # Run through conv blocks then classifier
        # squeeze removes the extra batch dimension so output is a plain number
        return self.classifier(self.features(x)).squeeze(1)


# Each CNN is just the base with its specific input channel count
class PRNUNet(_BaseCNN):
    # 2 channels: grayscale image + PRNU noise map
    def __init__(self): super().__init__(in_channels=2)

class ELANet(_BaseCNN):
    # 1 channel: ELA difference map
    def __init__(self): super().__init__(in_channels=1)

class FreqNet(_BaseCNN):
    # 3 channels: FFT map + DCT map + azimuthal map
    def __init__(self): super().__init__(in_channels=3)



# MODEL PATHS


SAVED_DIR = PROJECT_ROOT / "CNN" / "saved_modules"

# Two sets of models, one trained on general images one trained on faces
# the correct set is chosen based on image_type passed in from the server
MODEL_PATHS = {
    "general": {
        "prnu": SAVED_DIR / "prnu_general_v9.pth",
        "ela":  SAVED_DIR / "ela_general_v9.pth",
        "freq": SAVED_DIR / "freq_general_v9.pth",
    },
    "face": {
        "prnu": SAVED_DIR / "prnu_faces_v3.pth",
        "ela":  SAVED_DIR / "ela_faces_v3.pth",
        "freq": SAVED_DIR / "freq_faces_v3.pth",
    },
}

TARGET_SIZE = (224, 224)



# MODEL CACHE AND LOADER


# This dictionary lives in memory for the entire life of the server
# once a model is loaded it stays here so we never load it twice
_loaded_models: dict = {}

def _load_model(key, model_class, path, device):
    # Only load from disk if we have not loaded this model before
    if key not in _loaded_models:
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        # Create an empty model with the right architecture
        model = model_class()
        # Fill it with the trained weights from the saved file
        # map_location handles the case where model was trained on GPU
        # but is now running on a CPU machine
        model.load_state_dict(torch.load(path, map_location=device))
        # Move the model to the right device (GPU or CPU)
        model.to(device)
        # Switch to eval mode which disables dropout and changes
        # how batch normalization behaves, giving consistent predictions
        model.eval()
        # Store in cache so next call returns instantly
        _loaded_models[key] = model
        print(f"[Inference] Loaded {key} from {path.name}")

    return _loaded_models[key]


def get_device():
    # Use GPU if available, otherwise fall back to CPU
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_all_models():
    # Called once at server startup to pre load all six models
    # so the first real request does not have to wait for loading
    device = get_device()
    for image_type, paths in MODEL_PATHS.items():
        _load_model(f"prnu_{image_type}", PRNUNet, paths["prnu"], device)
        _load_model(f"ela_{image_type}",  ELANet,  paths["ela"],  device)
        _load_model(f"freq_{image_type}", FreqNet, paths["freq"], device)
    print("[Inference] All models loaded.")



# RUNNING THE CNN


# This decorator tells PyTorch not to track gradients here
# gradients are only needed during training, disabling them
# saves memory and makes inference faster
@torch.no_grad()
def _run_cnn(model, tensor, device) -> float:
    # The CNN expects a batch dimension as the first dimension
    # our single image is (channels, H, W)
    # unsqueeze(0) adds a batch of 1 making it (1, channels, H, W)
    inp = tensor.unsqueeze(0).to(device)

    # Run the image through the CNN, get back a raw logit
    logit = model(inp)

    # Sigmoid converts the raw logit into a probability between 0 and 1
    # 0 means the model is confident it is real
    # 1 means the model is confident it is AI generated
    # .item() converts the single element tensor into a plain Python float
    return float(torch.sigmoid(logit).item())


def _safe_cnn_score(key, model_class, path, tensor_fn,
                    image_path, device, is_freq=False):
    # Wraps the whole extraction and inference in a try except
    # if anything fails we return None instead of crashing the server
    # score fusion already knows how to handle None scores
    try:
        # build_frequency_tensor returns a tuple (tensor, radial_1d)
        # the other two return just a tensor
        # is_freq flag tells us which case we are in
        if is_freq:
            tensor, _ = tensor_fn(image_path, TARGET_SIZE)
        else:
            tensor = tensor_fn(image_path, TARGET_SIZE)

        model = _load_model(key, model_class, path, device)
        return _run_cnn(model, tensor, device)

    except Exception as e:
        print(f"[Inference] Warning {key} failed: {e}")
        return None



# MAIN PUBLIC FUNCTION


def run_inference(image_path: str, image_type: str = "general") -> dict:
    # This is what app.py calls when an image arrives at the server
    if image_type not in MODEL_PATHS:
        raise ValueError(f"image_type must be general or face, got {image_type}")

    device = get_device()
    paths  = MODEL_PATHS[image_type]

    # Run each CNN module, each one independently extracts its signal
    # and passes it through the trained network to get a 0 to 1 score
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

    # Metadata has no CNN, it just reads the file directly
    meta = analyze_metadata(image_path)

    # Return all four scores in one dictionary
    # this goes straight into fuse_scores in app.py
    return {
        "prnu_score":      prnu_score,
        "ela_score":       ela_score,
        "freq_score":      freq_score,
        "metadata_score":  meta["score"],
        "metadata_reason": meta["reason"],
        "metadata_format": meta.get("format", "UNKNOWN"),
        "image_type":      image_type,
    }