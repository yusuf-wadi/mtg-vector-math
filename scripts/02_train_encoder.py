"""Step 2 — Train the TF-IDF + MLP student encoder.

Loads the soft labels from Step 1 and trains a lightweight model:
  TF-IDF(oracle_text) → Linear(512) → ReLU → Linear(256) → ReLU → Linear(105) → Sigmoid

Loss: MSE against soft labels from bge-small teacher.

The trained weights are exported to pure numpy (CardEncoder) for
zero-dependency inference at deploy time.

Output:
  models/encoder.pkl   — CardEncoder, pure numpy, ~2-5MB
  models/axis_scale.npy — 99th-pct per-axis scale factors for global normalization
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from mtg_vec.encoder import build_torch_model, export_torch_to_numpy

DATA_DIR = _ROOT / "data"
MODELS_DIR = _ROOT / "models"
ORACLE_PATH = DATA_DIR / "oracle-slim.json.gz"
LABELS_PATH = DATA_DIR / "labels.npy"
INDEX_PATH = DATA_DIR / "card_index.json"

# Hyperparameters
EPOCHS = 30
BATCH_SIZE = 512
LR = 3e-4
WEIGHT_DECAY = 1e-5
VAL_SPLIT = 0.05
TFIDF_MAX_FEATURES = 12000
TFIDF_NGRAM_RANGE = (1, 2)   # unigrams + bigrams


def load_cards() -> list[dict]:
    with gzip.open(ORACLE_PATH, "rt", encoding="utf-8") as f:
        return json.load(f)


def card_text(card: dict) -> str:
    return f"{card.get('type_line', '')} {card.get('oracle_text', '')}".strip()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    print("Loading cards and labels...")
    cards = load_cards()
    labels = np.load(LABELS_PATH)  # (N, 105)
    with open(INDEX_PATH) as f:
        card_index = json.load(f)

    # Build text list in same order as labels
    card_map = {c.get("name"): c for c in cards}
    texts = [card_text(card_map.get(name, {})) for name in card_index]

    print(f"  {len(texts)} cards, {labels.shape[1]} axes")

    # Fit TF-IDF
    print(f"Fitting TF-IDF (max_features={TFIDF_MAX_FEATURES})...")
    tfidf = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        sublinear_tf=True,
        min_df=2,
    )
    X = tfidf.fit_transform(texts).toarray().astype(np.float32)  # (N, vocab)
    print(f"  TF-IDF shape: {X.shape}")

    # Train/val split
    N = len(X)
    val_n = max(1, int(N * VAL_SPLIT))
    idx = np.random.permutation(N)
    val_idx, train_idx = idx[:val_n], idx[val_n:]

    X_train = torch.from_numpy(X[train_idx]).to(device)
    y_train = torch.from_numpy(labels[train_idx]).to(device)
    X_val = torch.from_numpy(X[val_idx]).to(device)
    y_val = torch.from_numpy(labels[val_idx]).to(device)

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    # Build model
    model = build_torch_model(X.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.MSELoss()

    print(f"Training for {EPOCHS} epochs...")
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(Xb)
        train_loss /= len(train_idx)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val).item()

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS}  train={train_loss:.5f}  val={val_loss:.5f}")

    # Load best weights
    model.load_state_dict(best_state)
    print(f"Best val loss: {best_val_loss:.5f}")

    # Compute per-axis scale factors (99th pct across training set)
    model.eval()
    with torch.no_grad():
        all_preds = model(X_train.cpu()).numpy() if device.type == "cpu" else model(X_train).cpu().numpy()
    axis_scale = np.percentile(all_preds, 99, axis=0).astype(np.float32)
    np.save(MODELS_DIR / "axis_scale.npy", axis_scale)
    print(f"Saved axis_scale → {MODELS_DIR / 'axis_scale.npy'}")

    # Export to numpy CardEncoder
    MODELS_DIR.mkdir(exist_ok=True)
    enc = export_torch_to_numpy(model, tfidf, axis_scale=axis_scale)
    print("Done.")


if __name__ == "__main__":
    main()
