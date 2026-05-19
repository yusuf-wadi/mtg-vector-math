"""CardEncoder — TF-IDF + 2-layer MLP student model.

This is the deployed model: a TF-IDF vectorizer (sklearn) feeding a
two-layer MLP implemented in pure numpy. The full artifact is ~2-5MB
(sklearn TF-IDF vocab + two weight matrices) and runs without torch,
sentence-transformers, or any ML framework at inference time.

Training uses this module's MLP class with torch for gradient descent,
then exports the weights to numpy arrays for deployment.

Architecture:
  Input:  TF-IDF vector (vocab_size dims, typically ~8000-15000)
  Layer 1: Linear(vocab_size → 512) + ReLU + Dropout(0.1)
  Layer 2: Linear(512 → 256) + ReLU
  Output: Linear(256 → 105) + Sigmoid   [soft [0,1] per axis]

The sigmoid output matches the normalized soft-label targets from the
teacher (bge-small cosine similarities, clipped to [0,1]).
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np

_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "encoder.pkl"


# ---------------------------------------------------------------------------
# Numpy inference model (deployed)
# ---------------------------------------------------------------------------

class CardEncoder:
    """Loaded from models/encoder.pkl. Pure numpy inference.

    Usage:
        enc = CardEncoder.load()
        vec = enc.encode(oracle_text, type_line)   # → np.ndarray shape (105,)
        vecs = enc.encode_batch(texts)             # → np.ndarray shape (N, 105)
    """

    def __init__(
        self,
        tfidf,           # fitted sklearn TfidfVectorizer
        W1: np.ndarray,  # (vocab_size, 512)
        b1: np.ndarray,  # (512,)
        W2: np.ndarray,  # (512, 256)
        b2: np.ndarray,  # (256,)
        W3: np.ndarray,  # (256, 105)
        b3: np.ndarray,  # (105,)
        axis_scale: Optional[np.ndarray] = None,  # (105,) corpus 99th pct
    ):
        self.tfidf = tfidf
        self.W1, self.b1 = W1, b1
        self.W2, self.b2 = W2, b2
        self.W3, self.b3 = W3, b3
        self.axis_scale = axis_scale

    def _forward(self, X: np.ndarray) -> np.ndarray:
        """X: dense float32 (N, vocab_size). Returns (N, 105)."""
        h1 = np.maximum(0.0, X @ self.W1 + self.b1)   # ReLU
        h2 = np.maximum(0.0, h1 @ self.W2 + self.b2)  # ReLU
        logits = h2 @ self.W3 + self.b3
        return 1.0 / (1.0 + np.exp(-logits))           # Sigmoid → [0,1]

    def _card_text(self, oracle_text: str, type_line: str = "") -> str:
        return f"{type_line} {oracle_text}".strip()

    def encode(self, oracle_text: str, type_line: str = "") -> np.ndarray:
        """Encode a single card. Returns shape (105,)."""
        text = self._card_text(oracle_text, type_line)
        X = self.tfidf.transform([text]).toarray().astype(np.float32)
        return self._forward(X)[0]

    def encode_batch(
        self,
        cards: list[dict],  # list of {oracle_text, type_line}
    ) -> np.ndarray:
        """Encode a list of card dicts. Returns shape (N, 105)."""
        texts = [
            self._card_text(c.get("oracle_text", ""), c.get("type_line", ""))
            for c in cards
        ]
        X = self.tfidf.transform(texts).toarray().astype(np.float32)
        return self._forward(X)

    def save(self, path: Path = _MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=4)
        print(f"Saved encoder to {path} ({path.stat().st_size / 1e6:.1f} MB)")

    @classmethod
    def load(cls, path: Path = _MODEL_PATH) -> "CardEncoder":
        with open(path, "rb") as f:
            return pickle.load(f)


# ---------------------------------------------------------------------------
# PyTorch training model (offline only — not imported at inference time)
# ---------------------------------------------------------------------------

def build_torch_model(vocab_size: int):
    """Returns a torch.nn.Sequential for training.
    Import torch only here so the inference path stays torch-free."""
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(vocab_size, 512),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 105),
        nn.Sigmoid(),
    )


def export_torch_to_numpy(
    torch_model,
    tfidf,
    axis_scale: Optional[np.ndarray] = None,
    save_path: Path = _MODEL_PATH,
) -> CardEncoder:
    """Extract weights from a trained torch model into a CardEncoder."""
    import torch
    layers = [m for m in torch_model.modules() if hasattr(m, "weight")]
    # layers[0]=Linear(vocab→512), layers[1]=Linear(512→256), layers[2]=Linear(256→105)
    def w(layer): return layer.weight.detach().cpu().numpy().T.astype(np.float32)
    def b(layer): return layer.bias.detach().cpu().numpy().astype(np.float32)
    enc = CardEncoder(
        tfidf=tfidf,
        W1=w(layers[0]), b1=b(layers[0]),
        W2=w(layers[1]), b2=b(layers[1]),
        W3=w(layers[2]), b3=b(layers[2]),
        axis_scale=axis_scale,
    )
    enc.save(save_path)
    return enc
