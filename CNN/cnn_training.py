"""
cnn_training

learns via looking at small regions of the image
separated into three different CNNS: PRNU, ELA and freq

full training pipeline for three separate CNNs:
   PRNUNet  : input (2, 224, 224)  grayscale + PRNU residual
   ELANet   : input (1, 224, 224)  ELA map
   FreqNet  : input (3, 224, 224)  FFT + DCT + azimuthal maps

v2 updates:
   70% train, 15% val, 15% test split (test only evaluated at the end)
   Tuned dropout (0.3), gradient clipping


commands:
  python -m CNN.cnn_training --mode prnu
  python -m CNN.cnn_training --mode ela
  python -m CNN.cnn_training --mode freq
  python -m CNN.cnn_training --mode both
  python -m CNN.cnn_training --mode prnu --max_samples 50   <- smoke test
"""

import argparse
import random
import io
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report, roc_auc_score
from PIL import Image

# Your existing preprocessing functions
from .feature_extract.PRNU_and_ELA_preparing import build_prnu_tensor, build_ela_tensor
from .feature_extract.frequency_extractor import build_frequency_tensor


# Paths & global config


#general training path
#REAL_DIR = Path(r"C:\Users\rdc20\PycharmProjects\imageMetadata\images\training\general\real")
#AI_DIR   = Path(r"C:\Users\rdc20\PycharmProjects\imageMetadata\images\training\general\ai")

#faces training path
REAL_DIR = Path(r"C:\Users\rdc20\PycharmProjects\imageMetadata\images\training\faces\real")
AI_DIR   = Path(r"C:\Users\rdc20\PycharmProjects\imageMetadata\images\training\faces\ai")
IMAGE_EXTENSIONS  = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TARGET_SIZE       = (224, 224)

# Split ratios
TRAIN_SPLIT       = 0.70
VAL_SPLIT         = 0.15
TEST_SPLIT        = 0.15      # held out only evaluated after all epochs finish

BATCH_SIZE        = 64       # how many images are processed at the same time
NUM_EPOCHS        = 50       # number of times the modules sees the entire training set
LEARNING_RATE     = 3e-4      # how large of steps it takes when updating its weights
WEIGHT_DECAY      = 1e-4      # penalizing larger weight values
DROPOUT           = 0.3       #
GRAD_CLIP         = 1.0       # gradient clipping to prevent exploding gradients
SEED              = 42        # set random seed

# Saved models go into CNN/saved_modules/
SAVED_MODULES_DIR = Path(__file__).parent / "saved_modules"
SAVED_MODULES_DIR.mkdir(exist_ok=True)

PRNU_MODEL_PATH = SAVED_MODULES_DIR / "prnu_faces_v2.pth"
ELA_MODEL_PATH  = SAVED_MODULES_DIR / "ela_faces_v2.pth"
FREQ_MODEL_PATH = SAVED_MODULES_DIR / "freq_faces_v2.pth"



#   Augmentation helpers  (applied only during training)
# Applies random transformations to images before the model sees them
def augment_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    All operations preserve the tensor dtype and shape.

    Augmentations:
      - Random horizontal flip (50%)
      - Random vertical flip   (20%)
      - Gaussian noise         (50%)
      - Random resize + restore (30%)  - simulates social media compression
      - JPEG compression sim   (60%)   - only for single-channel ELA tensors
    """
    # Horizontal flip
    if random.random() < 0.5:
        tensor = torch.flip(tensor, dims=[2])

    # Vertical flip
    if random.random() < 0.2:
        tensor = torch.flip(tensor, dims=[1])

    # Gaussian noise added to prevent model from memorizing specific pixels
    if random.random() < 0.5:
        noise = torch.randn_like(tensor) * 0.02
        tensor = tensor + noise

    # Random resize + restore simulates image being scaled down and back up
    # happens when images are shared on social media or messaging apps
    if random.random() < 0.3:
        H, W = tensor.shape[1], tensor.shape[2]
        scale = random.uniform(0.7, 0.9)
        small = torch.nn.functional.interpolate(
            tensor.unsqueeze(0),
            scale_factor=scale,
            mode='bilinear',
            align_corners=False
        )
        tensor = torch.nn.functional.interpolate(
            small,
            size=(H, W),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

    # JPEG compression simulation for ELA (single channel)
    # Increased from 0.4 to 0.6. ELA is particularly sensitive to compression

    # PRNU frequencies don't need additional compression
    if tensor.shape[0] == 1 and random.random() < 0.6:
        try:
            arr = (tensor.squeeze(0).numpy() * 255).clip(0, 255).astype(np.uint8)
            pil = Image.fromarray(arr, mode='L')
            buf = io.BytesIO()
            quality = random.randint(60, 90)
            pil.save(buf, format='JPEG', quality=quality)
            buf.seek(0)
            compressed = np.array(Image.open(buf)).astype(np.float32) / 255.0
            tensor = torch.tensor(compressed[np.newaxis, ...], dtype=torch.float32)
        except Exception:
            pass

    return tensor



# Dataset

# Datasets variable from pytorch
class ImageForgeryDataset(Dataset):
    """
    Returns (tensor, label) pairs.
      label 0 -> real
      label 1 -> AI-generated

    mode       : "prnu" -> tensor shape (2, H, W)
                 "ela"  -> tensor shape (1, H, W)
                 "freq" -> tensor shape (3, H, W)
    max_samples: caps each class to this number (keeps dataset balanced)
    augment: if True, applies random augmentations (use only for train split)
    """

    def __init__(self,samples: list,mode: str = "prnu",augment: bool = False):
        self.samples = samples # All images file paths
        self.mode = mode # PRNU , ELA , freq
        self.augment = augment # True if training False if validation

    # num of samples
    def __len__(self):
        return len(self.samples)

    # Calls each individual sample
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            if self.mode == "prnu":
                tensor = build_prnu_tensor(str(path), TARGET_SIZE)
            elif self.mode == "ela":
                tensor = build_ela_tensor(str(path), TARGET_SIZE)
            else:  # freq
                tensor, _ = build_frequency_tensor(str(path), TARGET_SIZE)
        # If anything fails, skips
        except Exception as e:
            print(f"[WARN] Skipping {path.name}: {e}")
            c = 2 if self.mode == "prnu" else (1 if self.mode == "ela" else 3)
            tensor = torch.zeros(c, *TARGET_SIZE, dtype=torch.float32)

        if self.augment:
            tensor = augment_tensor(tensor)

        return tensor, torch.tensor(label, dtype=torch.float32)


def load_samples(real_dir: Path, ai_dir: Path,
                 max_samples: int = None) -> list:
    """Scans both folders recursively and returns a shuffled list of (path, label)"""



    real_files = [
        (p, 0) for p in real_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    ai_files = [
        (p, 1) for p in ai_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    # Caps classes on the same number to prevent a leading class
    if max_samples is not None:
        random.shuffle(real_files)
        random.shuffle(ai_files)
        real_files = real_files[:max_samples]
        ai_files   = ai_files[:max_samples]
        print(f"[Dataset] Capped to {len(real_files)} real + {len(ai_files)} AI")

    # Combine samples
    all_samples = real_files + ai_files
    random.shuffle(all_samples)
    print(f"[Dataset] Total samples: {len(all_samples)} "
          f"(real={len(real_files)}, ai={len(ai_files)})")
    return all_samples


def split_samples(samples: list) -> tuple:
    """
    Stratified split into train / val / test
    Splits each class separately then recombines so every split
    has the same real/AI ratio regardless of dataset size
    """
    real    = [s for s in samples if s[1] == 0]
    ai      = [s for s in samples if s[1] == 1]

    def split_class(lst):
        n = len(lst)
        n_train = int(n * TRAIN_SPLIT)
        n_val = int(n * VAL_SPLIT)
        return lst[:n_train], lst[n_train:n_train + n_val], lst[n_train + n_val:]

    r_train, r_val, r_test = split_class(real)
    a_train, a_val, a_test = split_class(ai)

    train = r_train + a_train
    val = r_val   + a_val
    test = r_test  + a_test

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    real_test = sum(1 for s in test if s[1] == 0)
    ai_test = sum(1 for s in test if s[1] == 1)
    print(f"[Split] train={len(train)} | val={len(val)} | test={len(test)} "
          f"(test: {real_test} real / {ai_test} AI)")
    return train, val, test



# CNN Architectures

# Building block for all networks to use
class ConvBlock(nn.Module):
    """Conv -> BN -> ReLU -> MaxPool (optional)

        layers:
            Conv2d: a small 3x3 filter, multiplies its weight with the corresponding pixels and sums them.
            this filter learns to detect things like edges corners, and deeper layers detect textures and structures

            BatchNorm2d: normalizes the output of the convolution (shifts large values to be small one around zero). ensures values flowing in the network to stay at a reasonable range
            also slightly reduces overfitting

            ReLU: function that introduces non-linearity

            MaxPool2d(2,2): takes the max value for each 2x2 region, reducing the spatial dimensions by half.
            this reduces the resolution while increasing the abstractions

    """



    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        layers = [
            # (no need to use bias)
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            # ReLu set to modify tensor
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)


    def forward(self, x):
        return self.block(x)


class PRNUNet(nn.Module):
    """
    Input : (B, 2, 224, 224)  -- grayscale + PRNU residual
    Output: (B,)  -- raw logit
    Spatial flow: 224 -> 112 -> 56 -> 28 -> 14 -> 7
    """


    def __init__(self, dropout: float = DROPOUT):
        super().__init__()
        # Increase the number of features maps the longer it goes on. more feature maps == more patterns cna be detected.
        # (feature maps refer to the grid produced by convolution)
        self.features = nn.Sequential(
            ConvBlock(2,   32),
            ConvBlock(32,  64),
            ConvBlock(64,  128),
            ConvBlock(128, 256),
            ConvBlock(256, 256),
        )

        # After the conv blocks dimensions are 7x7. flatten the map to convert into a 1d vector so it can work with CNN layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),

            # Drops out some neurons outputs to zero to learn redundant representation (not allowing it to rely on any single neuron)
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        # removes one dimension
        return self.classifier(self.features(x)).squeeze(1)

# (only difference between the nets is the conv block input channel)
class ELANet(nn.Module):
    """
    Input : (B, 1, 224, 224)  -- ELA map
    Output: (B,)              -- raw logit
    Spatial flow: 224 -> 112 -> 56 -> 28 -> 14 -> 7
    """

    def __init__(self, dropout: float = DROPOUT):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1,   32),
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


class FreqNet(nn.Module):
    """
    Input : (B, 3, 224, 224)  -- FFT map + DCT map + azimuthal map
    Output: (B,)              -- raw logit
    Spatial flow: 224 -> 112 -> 56 -> 28 -> 14 -> 7
    """

    def __init__(self, dropout: float = DROPOUT):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3,   32),
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



# Training & evaluation helpers

# Sets random seed for every number generator used in the project for consistency)
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Core training loop
def train_one_epoch(model, loader, criterion, optimizer, device):
    """
        Five steps are repeated for every batch:
            1 zero grad: clears gradients from the previous batch

            2 forward pass: runs the batch through the model to get predictions

            3 loss: calculates how wrong the predictions were using the loss function

            4 backwars: calculates how much weight contributed the error using backpropagation algorithm
            (backpropagation algorithm works backwards through the network layer by layer calculating the weight to find the error)

            5 clip grad norm: limits the total gradient magnitude to 1.0

            6 optimizer.step: updates all the weights in the direction that reduces loss

    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    # Gradients can decide if we should increase weight or not
    for tensors, labels in loader:
        tensors, labels = tensors.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(tensors)
        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping prevents exploding gradients
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        optimizer.step()

        total_loss += loss.item() * tensors.size(0)

        # converts the numbers into probabilities between zero and one
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += tensors.size(0)

    return total_loss / total, correct / total

# Do not compute gradients during evaluation
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0,0, 0
    all_probs, all_labels = [],[]

    for tensors, labels in loader:
        tensors, labels = tensors.to(device), labels.to(device)

        logits = model(tensors)
        loss = criterion(logits, labels)
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

        total_loss += loss.item() * tensors.size(0)
        correct += (preds == labels).sum().item()
        total += tensors.size(0)

        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy().astype(int))

    avg_loss = total_loss / total
    accuracy = correct / total

    # measures the modules ability to separate real from ai images. more informative than accuracy because its not affected by the choice of thresholds
    # AUC: Area Under the Roc curve (Receiver operating characteristic, graph that shows how the classifier performs)
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0

    return avg_loss, accuracy, auc, all_labels, [int(p >= 0.5) for p in all_probs]



# Main training function

def train(mode: str, max_samples: int = None):
    # Safety check
    assert mode in ("prnu", "ela", "freq")
    set_seed(SEED)

    # GPU check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 60}")
    print(f"  Training : {mode.upper()}Net")
    print(f"  Device  : {device}")
    if max_samples:
        print(f"  Mode   : SMOKE TEST  ({max_samples} images per class)")
    else:
        print(f" Mode   : FULL TRAINING (5k per class)")
    print(f"{'=' * 60}\n")

    # Load & split samples
    all_samples = load_samples(REAL_DIR, AI_DIR, max_samples=max_samples)
    train_s, val_s, test_s = split_samples(all_samples)

    # Datasets augmentation only on train
    train_ds = ImageForgeryDataset(train_s, mode=mode, augment=True)
    val_ds = ImageForgeryDataset(val_s,mode=mode, augment=False)
    test_ds = ImageForgeryDataset(test_s, mode=mode, augment=False)

    # batch size: Load 64 images
    # shuffle to randomize order
    # 4 workers processing at once
    # pin memory for faster cpu to gpu transfer
    # presistant workers to not kill workers between epochs
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=4, pin_memory=True, persistent_workers=True)
    # Model, loss, optimiser
    if mode == "prnu":
        model = PRNUNet().to(device)
    elif mode == "ela":
        model = ELANet().to(device)
    else:
        model = FreqNet().to(device)
    # Weighted loss: penalise AI misses more heavily.
    # pos_weight > 1 tells the loss to care more about the positive (AI) class.
    # Weight = real_count / ai_count so the two classes contribute equally.
    train_labels = [s[1] for s in train_s]
    n_real = train_labels.count(0)
    n_ai = train_labels.count(1)

    # used Binary Cross Entropy to calculate loss for binary classifications
    # measures how far the modules prediction is form the truth
    pos_w  = torch.tensor([n_real / max(n_ai, 1)], dtype=torch.float32).to(device)
    print(f"[Loss] pos_weight = {pos_w.item():.3f}  (real={n_real}, ai={n_ai})")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    # Uses adam W to adjust the learning rate of each individual weight based on gradient history
    optimizer = optim.AdamW(model.parameters(),lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Cosine annealing smoothly decays learning rate to near-zero over all epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    # Training loop
    # Only saves modules if there was improvment
    best_val_auc = 0.0
    if mode == "prnu":
        save_path = PRNU_MODEL_PATH
    elif mode == "ela":
        save_path = ELA_MODEL_PATH
    else:
        save_path = FREQ_MODEL_PATH

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_auc, _, _ = evaluate(
            model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch {epoch:>3}/{NUM_EPOCHS} | "
            f"Train  loss={train_loss:.4f}  acc={train_acc:.3f} | "
            f"Val  loss={val_loss:.4f}  acc={val_acc:.3f}  AUC={val_auc:.3f} | "
            f"LR={scheduler.get_last_lr()[0]:.2e}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), save_path)
            print(f"  -> Saved best model  (AUC {val_auc:.4f})")

    #  Final evaluation on held-out TEST set
    print(f"\n{'=' * 60}")
    print(f"  FINAL TEST SET EVALUATION  ({mode.upper()}Net)")
    print(f"{'=' * 60}")

    # Reload best saved weights before testing
    model.load_state_dict(torch.load(save_path, map_location=device))
    _, test_acc, test_auc, true_labels, pred_labels = evaluate(
        model, test_loader, criterion, device
    )

    print(classification_report(true_labels, pred_labels,
                                target_names=["Real", "AI"]))
    print(f"Test Accuracy : {test_acc:.4f}")
    print(f"Test AUC      : {test_auc:.4f}")
    print(f"Best Val AUC  : {best_val_auc:.4f}")
    print(f"Model saved   : {save_path}\n")



# 6.  Entry point

if __name__ == "__main__":

    # Pass to the command line arguments
    parser = argparse.ArgumentParser(description="Train PRNUNet and/or ELANet")
    parser.add_argument(
        "--mode",
        choices=["prnu", "ela", "freq", "both"],
        default="both",
        help="Which CNN to train: prnu, ela, freq, or both (default: both)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=10000,
        help="Images per class (default: 5000). Use --max_samples 50 for a smoke test."
    )
    args = parser.parse_args()

    if args.mode in ("prnu", "both"):
        train("prnu", max_samples=args.max_samples)
    if args.mode in ("ela", "both"):
        train("ela",  max_samples=args.max_samples)
    if args.mode in ("freq", "both"):
        train("freq", max_samples=args.max_samples)