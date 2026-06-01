"""
cnn_training.py

Full training pipeline for three separate CNNs that learn by looking at small regions of the image:
    PRNUNet input (2, 224, 224)  grayscale + PRNU residual
    ELANet input (1, 224, 224)  ELA map
    FreqNet input (3, 224, 224)  FFT + DCT + azimuthal maps


commands:
    python -m CNN.cnn_training --mode prnu
    python -m CNN.cnn_training --mode ela
    python -m CNN.cnn_training --mode freq
    python -m CNN.cnn_training --mode both
    python -m CNN.cnn_training --mode prnu --max_samples 50   <- smoke test
"""

import argparse # For script commands
import random
import io # Used for jpeg compression
from pathlib import Path # Cleaner way to work with paths, also adds auto linux compatibility

import numpy as np
import torch
import torch.nn as nn # Building blocks for CNN
import torch.optim as optim # optimize updating the model
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, roc_auc_score # measure how the model performs
from PIL import Image

from .feature_extract.PRNU_and_ELA_preparing import build_prnu_tensor, build_ela_tensor
from .feature_extract.frequency_extractor import build_frequency_tensor

# Paths

# General training paths
REAL_DIR = Path(r"D:\CNN_Training_Data\training\general\real")
AI_DIR   = Path(r"D:\CNN_Training_Data\training\general\ai")

# Face training paths (uncomment to switch)
# REAL_DIR = Path(r"D:\CNN_Training_Data\training\faces\real")
# AI_DIR   = Path(r"D:\CNN_Training_Data\training\faces\ai")

# Hyperparameters and global config

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Input image dimensions for all three CNNs
IMAGE_HEIGHT = 224
IMAGE_WIDTH  = 224
TARGET_SIZE  = (IMAGE_HEIGHT, IMAGE_WIDTH)

# Dataset split ratios
TRAIN_SPLIT = 0.70
VAL_SPLIT   = 0.15
TEST_SPLIT  = 0.15   # held out, only evaluated after all epochs finish

# Training hyperparameters
BATCH_SIZE    = 64     # number of images processed in one forward pass
NUM_EPOCHS    = 50     # number of full passes over the training set
LEARNING_RATE = 3e-4   # step size for weight updates
WEIGHT_DECAY  = 1e-4   # L2 penalty to discourage large weights
DROPOUT       = 0.3    # fraction of neurons randomly dropped during training
GRAD_CLIP     = 1.0    # maximum gradient norm (prevents exploding gradients)
SEED          = 42     # random seed for reproducibility

# DataLoader worker count
NUM_WORKERS = 4

# Augmentation probabilities
HFLIP_PROB         = 0.5   # random horizontal flip
VFLIP_PROB         = 0.2   # random vertical flip
NOISE_PROB         = 0.5   # Gaussian noise injection
RESIZE_PROB        = 0.3   # random resize + restore (simulates compression)
JPEG_COMPRESS_PROB = 0.6   # JPEG compression simulation (ELA channel only)

# JPEG augmentation quality range
JPEG_QUALITY_MIN = 60
JPEG_QUALITY_MAX = 90

# Random resize scale range (simulates social-media downscaling)
RESIZE_SCALE_MIN = 0.7
RESIZE_SCALE_MAX = 0.9

# Gaussian noise standard deviation
GAUSSIAN_NOISE_STD = 0.02

# Saved models directory
SAVED_MODULES_DIR = Path(__file__).parent / "saved_modules"
SAVED_MODULES_DIR.mkdir(exist_ok=True)

PRNU_MODEL_PATH = SAVED_MODULES_DIR / "prnu_general_v8.pth"
ELA_MODEL_PATH  = SAVED_MODULES_DIR / "ela_general_v8.pth"
FREQ_MODEL_PATH = SAVED_MODULES_DIR / "freq_general_v8.pth"

# Face model paths (uncomment to switch)
# PRNU_MODEL_PATH = SAVED_MODULES_DIR / "prnu_faces_v2.pth"
# ELA_MODEL_PATH  = SAVED_MODULES_DIR / "ela_faces_v2.pth"
# FREQ_MODEL_PATH = SAVED_MODULES_DIR / "freq_faces_v2.pth"

# Classification threshold
DECISION_THRESHOLD = 0.5


# Augmentation helpers (applied only during training)
# Randomly transform an image slightly every time its loaded so that the model does not see the same image twice (only applied in training)
def augment_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Applies random augmentations to a tensor. All operations preserve dtype and shape.

    Augmentations:
        Random horizontal flip  (HFLIP_PROB)
        Random vertical flip    (VFLIP_PROB)
        Gaussian noise          (NOISE_PROB)
        Random resize + restore (RESIZE_PROB)   simulates social-media compression
        JPEG compression sim    (JPEG_COMPRESS_PROB)  only for single-channel ELA tensors
    """

    #happens 50% of the time
    # Horizontal flip
    if random.random() < HFLIP_PROB:
        tensor = torch.flip(tensor, dims=[2])

    # Vertical flip , happens 20% of the time
    if random.random() < VFLIP_PROB:
        tensor = torch.flip(tensor, dims=[1])

    # Gaussian noise prevents the model from memorizing specific pixel patterns, happens 50% of the time
    if random.random() < NOISE_PROB:
        noise  = torch.randn_like(tensor) * GAUSSIAN_NOISE_STD # creates a tensor of the same shape filled with gaussian noise and apply it to the image
        tensor = tensor + noise

    # Random resize + restore simulates an image being scaled down and back up,
    # as happens when images are shared on social media or messaging apps
    # happens 30% of the time
    if random.random() < RESIZE_PROB:
        H, W  = tensor.shape[1], tensor.shape[2]
        scale = random.uniform(RESIZE_SCALE_MIN, RESIZE_SCALE_MAX) # random number between 70% and 90%
        small = torch.nn.functional.interpolate(
            tensor.unsqueeze(0),
            scale_factor=scale,
            mode='bilinear',
            align_corners=False,
        )
        tensor = torch.nn.functional.interpolate(
            small,
            size=(H, W),
            mode='bilinear', # uses surrounding pixels to determine value
            align_corners=False,
        ).squeeze(0)

    # JPEG compression simulation only for single-channel (ELA) tensors.
    # ELA is particularly sensitive to compression artifacts.
    # Gives the image more compression history to improve ELA module
    if tensor.shape[0] == 1 and random.random() < JPEG_COMPRESS_PROB:
        try:
            arr     = (tensor.squeeze(0).numpy() * 255).clip(0, 255).astype(np.uint8)
            pil     = Image.fromarray(arr, mode='L')
            buf     = io.BytesIO()
            quality = random.randint(JPEG_QUALITY_MIN, JPEG_QUALITY_MAX)
            pil.save(buf, format='JPEG', quality=quality)
            buf.seek(0)
            compressed = np.array(Image.open(buf)).astype(np.float32) / 255.0
            tensor     = torch.tensor(compressed[np.newaxis, ...], dtype=torch.float32)
        except Exception:
            pass

    return tensor


# Dataset

# Stores a list of (Path, lable)
class ImageForgeryDataset(Dataset):
    """
    Returns (tensor, label) pairs.
        label 0 -> real
        label 1 -> AI-generated

    mode       : "prnu" -> tensor shape (2, H, W)
                 "ela"  -> tensor shape (1, H, W)
                 "freq" -> tensor shape (3, H, W)
    augment    : if True, applies random augmentations (use only for train split)
    """

    def __init__(self, samples: list, mode: str = "prnu", augment: bool = False):
        self.samples = samples   # list of (path, label) tuples
        self.mode    = mode      # which feature extractor to use
        self.augment = augment   # True during training, False during val/test

    # Tells the dataset how many samples exist
    def __len__(self):
        return len(self.samples)


    # Called every time the CNN needs a sample, receives an index and returns the path and label
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            if self.mode == "prnu":
                tensor = build_prnu_tensor(str(path), TARGET_SIZE)
            elif self.mode == "ela":
                tensor = build_ela_tensor(str(path), TARGET_SIZE)
            else:  # freq
                tensor, _ = build_frequency_tensor(str(path), TARGET_SIZE)
        except Exception as e:
            print(f"[WARN] Skipping {path.name}: {e}")
            # Number of input channels per mode
            channels_by_mode = {"prnu": 2, "ela": 1, "freq": 3}
            c      = channels_by_mode.get(self.mode, 1)
            tensor = torch.zeros(c, *TARGET_SIZE, dtype=torch.float32)

        if self.augment:
            tensor = augment_tensor(tensor) # Applies data augmentation

        return tensor, torch.tensor(label, dtype=torch.float32)


# Data loading helpers

def load_samples(real_dir: Path, ai_dir: Path, max_samples: int = None) -> list:
    """Scans both folders recursively and returns a shuffled list of (path, label)."""
    # Checks every file and labels it
    real_files = [
        (p, 0) for p in real_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    ai_files = [
        (p, 1) for p in ai_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


    # Cap both classes to the same count to keep the dataset balanced
    # Ensure module sees an equal amount of real and fake images
    if max_samples is not None:
        random.shuffle(real_files)
        random.shuffle(ai_files)
        real_files = real_files[:max_samples]
        ai_files   = ai_files[:max_samples]
        print(f"[Dataset] Capped to {len(real_files)} real + {len(ai_files)} AI")

    # Combine the list and shuffle them
    all_samples = real_files + ai_files
    random.shuffle(all_samples)
    print(f"[Dataset] Total samples: {len(all_samples)} "
          f"(real={len(real_files)}, ai={len(ai_files)})")
    return all_samples


def split_samples(samples: list) -> tuple:
    """
    Stratified split into train / val / test sets.
    Splits each class separately then recombines so every split
    has the same real/AI ratio regardless of dataset size.
    """

    # Makes sure the split is equal between real and ai
    real = [s for s in samples if s[1] == 0]
    ai   = [s for s in samples if s[1] == 1]

    # train validation and testing splits
    # splits into three classes
    def split_class(lst):
        n       = len(lst)
        n_train = int(n * TRAIN_SPLIT)
        n_val   = int(n * VAL_SPLIT)
        return lst[:n_train], lst[n_train:n_train + n_val], lst[n_train + n_val:]

    r_train, r_val, r_test = split_class(real)
    a_train, a_val, a_test = split_class(ai)

    train = r_train + a_train
    val   = r_val   + a_val
    test  = r_test  + a_test
    # Shuffles again
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    real_test = sum(1 for s in test if s[1] == 0)
    ai_test   = sum(1 for s in test if s[1] == 1)
    print(f"[Split] train={len(train)} | val={len(val)} | test={len(test)} "
          f"(test: {real_test} real / {ai_test} AI)")

    return train, val, test


# CNN architectures

class ConvBlock(nn.Module):
    """
    Building block shared by all three networks.
    Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d (optional)

    Conv2d       : a 3x3 filter that learns to detect edges, corners, textures, and structures.
    BatchNorm2d  : normalizes conv output to keep values in a reasonable range; reduces overfitting.
    ReLU         : introduces non-linearity.
    MaxPool2d    : takes the max of each 2x2 region, halving spatial dimensions while increasing abstraction.
    """

    def __init__(self, in_ch: int, out_ch: int, pool: bool = True):
        super().__init__()

        # Define the conv blocks
        layers = [
            # slides a 3x3 grid of learnable weights
            # in_ch controls how many inputs and out_ch is outputs
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False), # padding adds a border of zeros around the input

            # normalizes the values
            nn.BatchNorm2d(out_ch),
            # Applies the ReLu unfction, and replaces every negative value with zero
            # ReLu also introduces non linearity that allows the CNN to learn complex patterns
            nn.ReLU(inplace=True), # modifies the tensor directly
        ]
        if pool:
            # Divides the feature maps into a 2x2 region and only keeps max values for each region
            # Helps with computing the layers
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers) # Flattens from a 2D to a 1D


    def forward(self, x):
        return self.block(x)


# Flattened feature size after 5 ConvBlocks on a 224x224 input:
#   224 -> 112 -> 56 -> 28 -> 14 -> 7  (each MaxPool halves dimensions)
#   final spatial size = 7x7, last channel count = 256
CNN_FINAL_CHANNELS   = 256
CNN_FINAL_SPATIAL    = 7
CNN_FLAT_FEATURES    = CNN_FINAL_CHANNELS * CNN_FINAL_SPATIAL * CNN_FINAL_SPATIAL  # 12544

# Fully-connected hidden layer sizes
FC_HIDDEN_LARGE = 512
FC_HIDDEN_SMALL = 128


class PRNUNet(nn.Module):
    """
    Input : (B, 2, 224, 224) -- grayscale + PRNU residual
    Output: (B,)             -- raw logit
    Spatial flow: 224 -> 112 -> 56 -> 28 -> 14 -> 7
    """

    PRNU_IN_CHANNELS = 2

    def __init__(self, dropout: float = DROPOUT):
        super().__init__()
        """
        conv block layout
        
        each conv layer has x amount of filters, each filter has its own weights and comes to its own conclusions
        the filter has weights on each point of it, each weight is multiplied by the pixel at the same position as it
        then the values are sumed up, and the filter moves one position.
        then the map thats created is given to the next conv block.
        
        the next conv block goes over all of the maps from the prev one at the same position at the same time with the same 3x3 filter
        then it makes a map of its own
        
        Input image         = (2, 224, 224)    your 2 channel PRNU tensor
            First conv + pool   = (32, 112, 112)   32 separate maps, half the size
            Second conv + pool  = (64, 56, 56)     64 separate maps, half again
            Third conv + pool   = (128, 28, 28)    128 separate maps
            Fourth conv + pool  = (256, 14, 14)    256 separate maps
            Fifth conv + pool   = (256, 7, 7)      256 separate maps, tiny spatial size
            Flatten             = (12544,)         all maps unrolled into one long vector
            Fully connected     = final decision   real or AI
            
        """
        # the actual pattern detector
        self.features = nn.Sequential(
            ConvBlock(self.PRNU_IN_CHANNELS, 32), # 32 feature maps
            ConvBlock(32,  64),             # 64 feature maps
            ConvBlock(64,  128),            # 128 feature maps
            ConvBlock(128, CNN_FINAL_CHANNELS),    # 256
            ConvBlock(CNN_FINAL_CHANNELS, CNN_FINAL_CHANNELS), # 256
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), # Converts the (256,7,7) tensor into a 12544 vector

            nn.Linear(CNN_FLAT_FEATURES, FC_HIDDEN_LARGE),
            nn.ReLU(inplace=True),  # First fully connected layer continues to lower values to 512
            nn.Dropout(dropout), # Dropout removes some of the neurons

            nn.Linear(FC_HIDDEN_LARGE, FC_HIDDEN_SMALL), # second fully connected layer goes to 128 values
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),


            nn.Linear(FC_HIDDEN_SMALL, 1), # Final layer, produces final result
        )

    def forward(self, x):
        return self.classifier(self.features(x)).squeeze(1) # squeeze to meet loss function demand


class ELANet(nn.Module):
    """
    Input : (B, 1, 224, 224) -- ELA map
    Output: (B,)             -- raw logit
    Spatial flow: 224 -> 112 -> 56 -> 28 -> 14 -> 7
    """

    ELA_IN_CHANNELS = 1

    def __init__(self, dropout: float = DROPOUT):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(self.ELA_IN_CHANNELS, 32),
            ConvBlock(32,  64),
            ConvBlock(64,  128),
            ConvBlock(128, CNN_FINAL_CHANNELS),
            ConvBlock(CNN_FINAL_CHANNELS, CNN_FINAL_CHANNELS),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(CNN_FLAT_FEATURES, FC_HIDDEN_LARGE),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(FC_HIDDEN_LARGE, FC_HIDDEN_SMALL),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(FC_HIDDEN_SMALL, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x)).squeeze(1)


class FreqNet(nn.Module):
    """
    Input : (B, 3, 224, 224) -- FFT map + DCT map + azimuthal map
    Output: (B,)             -- raw logit
    Spatial flow: 224 -> 112 -> 56 -> 28 -> 14 -> 7
    """

    FREQ_IN_CHANNELS = 3

    def __init__(self, dropout: float = DROPOUT):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(self.FREQ_IN_CHANNELS, 32),
            ConvBlock(32,  64),
            ConvBlock(64,  128),
            ConvBlock(128, CNN_FINAL_CHANNELS),
            ConvBlock(CNN_FINAL_CHANNELS, CNN_FINAL_CHANNELS),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(CNN_FLAT_FEATURES, FC_HIDDEN_LARGE),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(FC_HIDDEN_LARGE, FC_HIDDEN_SMALL),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(FC_HIDDEN_SMALL, 1),
        )
    # Runs the conv blocks
    def forward(self, x):
        return self.classifier(self.features(x)).squeeze(1)


# Training and evaluation helpers
"""
gradients tell us how much changing a specific weight will effect the loss
the less gradients and the less loss the better

steps that training takes
1 zero_grad     --> clear old gradients
2 forward pass  --> run images through model, get predictions
3 loss          --> measure how wrong the predictions were
4 backward      --> calculate how much each weight contributed to the error
5 clip grads    --> prevent any single update from being too large
6 optimizer     --> update all weights in the direction that reduces loss


"""
def set_seed(seed: int):
    """Sets random seed across all number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    adamw basicly = new weight = old weight - (learning rate * gradient)
    slightly shrink weight every update so one weight does not dominate


    Runs one full pass over the training set. For each batch:
        1. zero_grad   : clears gradients from the previous batch
        2. forward     : runs the batch through the model to get predictions
        3. loss        : measures how wrong the predictions are -- loss = -[label x log(prob) + (1 - label) x log(1 - prob)] label is the correct or incorrect result
        4. backward    : computes how much each weight contributed to the error (backpropagation)
        5. clip_grad   : caps gradient magnitude to GRAD_CLIP to prevent exploding gradients
        6. step        : updates all weights in the direction that reduces loss
    """

    # Switches into training mode
    model.train() # in training mode randomly zeros out 30% on neurons
    total_loss, correct, total = 0.0, 0, 0 # counters


    # Core of the CNN where weights are updated
    # one batch at a tome each batch is 64
    for tensors, labels in loader:
        tensors, labels = tensors.to(device), labels.to(device)

        optimizer.zero_grad() # clear gradients from the previous batch

        # Runs the batch though all five conv blocks, preduces a number for each image
        logits = model(tensors)
        loss   = criterion(logits, labels) # applies a sigmoid and then calculates how wrong each prediction was

        loss.backward() # Starts the backwards progression

        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP) # checks gradients and scales them if they are too big
        # and send extreme gradients back to training

        # updates all the weights
        # adamw optimizes updating the weights giving weights that point towards the same direction more value
        optimizer.step()

        total_loss += loss.item() * tensors.size(0) # grants us the average loss for the 64 images and converts it to total
        preds       = (torch.sigmoid(logits) >= DECISION_THRESHOLD).float() # converts into 0 and 1, one true zero false
        correct    += (preds == labels).sum().item() # counts the correct guesses
        total      += tensors.size(0)

    return total_loss / total, correct / total # calculates acuracy based on the loss


@torch.no_grad() # Tells pytourch not to track operations for gradient calculations
# we do not need to improve the model here so no need to track it
def evaluate(model, loader, criterion, device):
    """Evaluates the model without computing gradients."""
    model.eval() # switches to eval mode
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels      = [], []

    # same foward pass as training but with no backwards progression or anything for training, just evaluation
    for tensors, labels in loader:
        tensors, labels = tensors.to(device), labels.to(device)

        logits = model(tensors)
        loss   = criterion(logits, labels)
        probs  = torch.sigmoid(logits)
        preds  = (probs >= DECISION_THRESHOLD).float()

        total_loss += loss.item() * tensors.size(0)
        correct    += (preds == labels).sum().item()
        total      += tensors.size(0)

        all_probs.extend(probs.cpu().numpy()) # Collect all lables for evaluation
        all_labels.extend(labels.cpu().numpy().astype(int))

    avg_loss = total_loss / total
    accuracy = correct / total

    # AUC (Area Under the ROC Curve) measures ability to separate real from AI images.
    # More informative than accuracy because it is threshold independent.

    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0

    return avg_loss, accuracy, auc, all_labels, [int(p >= DECISION_THRESHOLD) for p in all_probs]


# Main training function

# ties all the other functions together and starts the training
def train(mode: str, max_samples: int = None):
    assert mode in ("prnu", "ela", "freq") # check mode

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'=' * 60}")
    print(f"  Training: {mode.upper()}Net")
    print(f"  Device: {device}")
    if max_samples:
        print(f"  Mode: SMOKE TEST  ({max_samples} images per class)")
    else:
        print(f"  Mode: FULL TRAINING")
    print(f"{'=' * 60}\n")

    # Load and split data
    all_samples           = load_samples(REAL_DIR, AI_DIR, max_samples=max_samples) # scans and loads both folders
    train_s, val_s, test_s = split_samples(all_samples) # does the sample split from before

    # Build datasets (augmentation only on the training split)
    # the augment is for creating inconsistencies useful for training only
    train_ds = ImageForgeryDataset(train_s, mode=mode, augment=True)
    val_ds   = ImageForgeryDataset(val_s,   mode=mode, augment=False)
    test_ds  = ImageForgeryDataset(test_s,  mode=mode, augment=False)

    # handles data delivery
    # Bundles 64 images into one batch and the model processes them all at once on the GPU
    # number of workers defines the amount of processes on the gpu that work on processing the images
    # the cpu also prepares the next batch with its own 4 workers in between runs
    # pin memory stops from saving the images to the disk and allows the gpu to run faster
    # persistent workers causes the process to not close in between batches
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,)
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,)
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,)

    # Select model
    if mode == "prnu":
        model = PRNUNet().to(device)
    elif mode == "ela":
        model = ELANet().to(device)
    else:
        model = FreqNet().to(device)

    # Weighted BCE loss: balances the real and AI classes automatically.
    # If some data is missing == give more weight to the side with less images
    # pos_weight = real_count / ai_count so both classes contribute equally to the loss.
    train_labels = [s[1] for s in train_s]
    n_real = train_labels.count(0)
    n_ai   = train_labels.count(1)
    pos_w  = torch.tensor([n_real / max(n_ai, 1)], dtype=torch.float32).to(device) # loss function
    print(f"[Loss] pos_weight = {pos_w.item():.3f}  (real={n_real}, ai={n_ai})")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    # AdamW adjusts the learning rate of each weight individually based on gradient history
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,) # automatically adjust learning rate


    # Controls the learning rate
    # Cosine annealing smoothly decays the learning rate to near-zero over all epochs
    MIN_LR    = 1e-6
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=MIN_LR,)

    # Training loop  saves the model only when validation AUC improves
    best_val_auc = 0.0
    save_path    = {"prnu": PRNU_MODEL_PATH, "ela": ELA_MODEL_PATH, "freq": FREQ_MODEL_PATH}[mode]

    # Each epoch runs one full pass through the training set then evaluates on the validation set
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,)
        val_loss, val_acc, val_auc, _, _ = evaluate(model, val_loader, criterion, device,)
        scheduler.step()

        # Summery
        print(
            f"Epoch {epoch:>3}/{NUM_EPOCHS} | "
            f"Train  loss={train_loss:.4f}  acc={train_acc:.3f} | "
            f"Val  loss={val_loss:.4f}  acc={val_acc:.3f}  AUC={val_auc:.3f} | "
            f"LR={scheduler.get_last_lr()[0]:.2e}")
        # Where the model is saved
        # Only save model when AUC improves
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved best model  (AUC {val_auc:.4f})")

    # Final evaluation on the held-out test set using the best saved weights
    print(f"\n{'=' * 60}")
    print(f"  FINAL TEST SET EVALUATION  ({mode.upper()}Net)")
    print(f"{'=' * 60}")


    # runs on the test set once
    model.load_state_dict(torch.load(save_path, map_location=device))
    _, test_acc, test_auc, true_labels, pred_labels = evaluate(
        model, test_loader, criterion, device,)

    print(classification_report(true_labels, pred_labels, target_names=["Real", "AI"]))
    print(f"Test Accuracy : {test_acc:.4f}")
    print(f"Test AUC      : {test_auc:.4f}")
    print(f"Best Val AUC  : {best_val_auc:.4f}")
    print(f"Model saved   : {save_path}\n")


# Entry point

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PRNUNet, ELANet, and/or FreqNet")

    # Runs the CNN command line
    parser.add_argument(
        "--mode",
        choices=["prnu", "ela", "freq", "both"],
        default="both",
        help="Which CNN to train: prnu, ela, freq, or both (default: both)",)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=10000,
        help="Images per class (default: 10000). Use --max_samples 50 for a smoke test.",)
    args = parser.parse_args()

    # start up the training
    if args.mode in ("prnu", "both"):
        train("prnu", max_samples=args.max_samples)
    if args.mode in ("ela", "both"):
        train("ela",  max_samples=args.max_samples)
    if args.mode in ("freq", "both"):
        train("freq", max_samples=args.max_samples)
