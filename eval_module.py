"""
eval_model.py
-------------
Evaluates a saved model on a test set without retraining.
Useful for checking scores after training logs have been lost.

Place at:  imageMetadata/eval_model.py
Run with:
  python eval_model.py --mode prnu
  python eval_model.py --mode ela
  python eval_model.py --mode freq
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import classification_report

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CNN.cnn_training import (
    PRNUNet, ELANet, FreqNet,
    ImageForgeryDataset, load_samples, split_samples,
    evaluate, set_seed,
    REAL_DIR, AI_DIR, SAVED_MODULES_DIR,
    BATCH_SIZE, SEED,
)
from torch.utils.data import DataLoader

MODELS = {
    "prnu": (PRNUNet, SAVED_MODULES_DIR / "prnu_v3.pth"),
    "ela":  (ELANet,  SAVED_MODULES_DIR / "ela_v3.pth"),
    "freq": (FreqNet, SAVED_MODULES_DIR / "freq_v3.pth"),
}


def eval_model(mode: str, max_samples: int = 10000):
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_class, model_path = MODELS[mode]

    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        return

    print(f"\nEvaluating {mode.upper()}Net from {model_path.name}")
    print(f"Device: {device}\n")

    all_samples   = load_samples(REAL_DIR, AI_DIR, max_samples=max_samples)
    _, _, test_s  = split_samples(all_samples)

    test_ds     = ImageForgeryDataset(test_s, mode=mode, augment=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0, pin_memory=True)

    model = model_class().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    criterion = nn.BCEWithLogitsLoss()
    _, test_acc, test_auc, true_labels, pred_labels = evaluate(
        model, test_loader, criterion, device
    )

    print(classification_report(true_labels, pred_labels,
                                 target_names=["Real", "AI"]))
    print(f"Test Accuracy : {test_acc:.4f}")
    print(f"Test AUC      : {test_auc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["prnu", "ela", "freq"], required=True)
    parser.add_argument("--max_samples", type=int, default=10000)
    args = parser.parse_args()
    eval_model(args.mode, args.max_samples)