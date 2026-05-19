"""Deck-level operations: centroid, card overlay, radar shape.

A deck's identity in the 105D space is its card centroid —
the mean of all card vectors, weighted by quantity.

Card overlay: given a deck centroid and a candidate card vector,
compute how the card shifts the centroid and what that means
for each radar spoke.
"""
from __future__ import annotations

import numpy as np
from typing import Optional

from .axes import (
    AXIS_IDS,
    FAMILIES,
    AXIS_INDEX,
    normalize_local,
    normalize_global,
    radar_collapse,
    radar_winning_axis,
)


def deck_centroid(
    card_vectors: np.ndarray,   # shape (N, 105)
    quantities: Optional[np.ndarray] = None,  # shape (N,), default all 1s
) -> np.ndarray:
    """Weighted mean of card vectors. Returns shape (105,)."""
    if quantities is None:
        quantities = np.ones(len(card_vectors), dtype=np.float32)
    quantities = quantities.astype(np.float32)
    return np.average(card_vectors, axis=0, weights=quantities)


def card_overlay(
    centroid: np.ndarray,     # deck centroid, shape (105,)
    card_vec: np.ndarray,     # candidate card vector, shape (105,)
    axis_scale: Optional[np.ndarray] = None,
) -> dict:
    """Compute how a candidate card relates to a deck's centroid.

    Returns a dict with:
      - card_radar: 12-spoke radar for the card alone
      - deck_radar: 12-spoke radar for the deck centroid
      - delta_radar: card - deck per spoke (positive = card strengthens)
      - cosine_sim: how aligned the card is with the deck's direction
      - top_axes: top 5 axes where the card contributes most
      - gap_axes: top 5 axes where the deck is weakest and card helps
    """
    # Normalize both to the same scale for comparison
    if axis_scale is not None:
        card_norm = normalize_global(card_vec, axis_scale)
        deck_norm = normalize_global(centroid, axis_scale)
    else:
        card_norm = normalize_local(card_vec)
        deck_norm = normalize_local(centroid)

    card_radar = radar_collapse(card_norm)
    deck_radar = radar_collapse(deck_norm)

    delta_radar = {
        fam: card_radar[fam] - deck_radar[fam]
        for fam in card_radar
    }

    # Cosine similarity between card vector and deck centroid (raw, not normalized)
    denom = (np.linalg.norm(card_vec) * np.linalg.norm(centroid))
    cosine_sim = float(np.dot(card_vec, centroid) / denom) if denom > 1e-9 else 0.0

    # Top axes where card contributes most
    top_axes = [
        {"axis": AXIS_IDS[i], "score": float(card_norm[i])}
        for i in np.argsort(card_norm)[::-1][:5]
    ]

    # Gap axes: deck is weak (bottom quartile) but card is strong
    deck_weak = deck_norm < np.percentile(deck_norm, 25)
    card_strong = card_norm > np.percentile(card_norm, 75)
    gap_mask = deck_weak & card_strong
    gap_axes = [
        {"axis": AXIS_IDS[i], "score": float(card_norm[i])}
        for i in np.where(gap_mask)[0]
    ]

    return {
        "card_radar": card_radar,
        "deck_radar": deck_radar,
        "delta_radar": delta_radar,
        "cosine_sim": cosine_sim,
        "top_axes": top_axes,
        "gap_axes": gap_axes,
        "card_winning_axes": radar_winning_axis(card_norm),
    }


def deck_shape_summary(centroid: np.ndarray, axis_scale: Optional[np.ndarray] = None) -> dict:
    """Human-readable summary of a deck's 12-spoke shape."""
    if axis_scale is not None:
        norm = normalize_global(centroid, axis_scale)
    else:
        norm = normalize_local(centroid)

    radar = radar_collapse(norm)
    winning = radar_winning_axis(norm)
    family_labels = {fam: data["label"] for fam, data in FAMILIES.items()}

    spokes = sorted(radar.items(), key=lambda x: x[1], reverse=True)
    return {
        "radar": radar,
        "winning_axes": winning,
        "top_families": [
            {
                "family": fam,
                "label": family_labels[fam],
                "score": score,
                "winning_axis": winning[fam],
            }
            for fam, score in spokes[:5]
        ],
    }
