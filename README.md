# mtg-vector-math

Lightweight MTG card embedding system. Maps any card's oracle text into a **105-dimensional axis space** derived from MTG's rules-vocabulary primitives, then collapses to a **12-spoke radar** for human-readable deck identity visualization.

## Architecture

```
Phase 1 — Label generation (offline)
  oracle_text
    → bge-small-en-v1.5 embeddings (teacher)
    → cosine_sim(axis_seed_embeddings)
    → 105D soft labels  [richer than regex alone]

Phase 2 — Student training (offline)
  oracle_text
    → TF-IDF features
    → 2-layer MLP
    → 105D vector
  Loss: MSE against Phase 1 soft labels

Phase 3 — Deployment (Vercel / anywhere)
  saved TF-IDF vocab + MLP weights (~2–5MB)
  pure numpy inference, zero ML framework dependency
  cold start: <100ms
```

## Why this design

- **Teacher (bge-small):** Understands novel wording, metaphor, indirect references. Generalizes to new card sets out of the box.
- **Student (TF-IDF + MLP):** ~2MB artifact. Runs on Vercel serverless with zero ML framework. Inference is a matrix multiply.
- **105 axes:** Defined in `mtg_vec/axes.py`, seeded from `data/mtg-axes.json`. Each axis is one dimension of the embedding space. The 12-family radar is a lossy but human-readable projection.
- **Deck identity = cluster centroid:** A deck's position in 105D space is the mean of its cards' vectors. Card overlay = distance from the deck centroid, visualized on the radar.

## Validation — king/queen test

A good embedding satisfies:
```
vec("Llanowar Elves") - vec("creature") + vec("instant") ≈ vec("Elvish Spirit Guide")
vec("Brainstorm") - vec("draw") + vec("discard") ≈ vec("Faithless Looting")
centroid(Sram Equipment) - equipment + aura ≈ centroid(Bruna Aura Voltron)
```

## Repo structure

```
mtg-vector-math/
├── data/                        # card data (gitignored if large)
│   ├── mtg-axes.json            # 105 axis seed definitions
│   └── oracle-slim.json.gz      # ~30k cards, oracle text + type line
├── scripts/
│   ├── 01_generate_labels.py    # bge-small teacher → 105D soft labels
│   ├── 02_train_encoder.py      # train TF-IDF + MLP student
│   └── 03_validate.py           # king/queen, centroid separation, kNN
├── mtg_vec/
│   ├── __init__.py
│   ├── axes.py                  # axis definitions, family map, normalization
│   ├── encoder.py               # TF-IDF + MLP model, save/load
│   └── deck.py                  # deck centroid, card overlay, radar collapse
├── models/                      # saved weights (gitignored)
│   └── .gitkeep
├── requirements.txt
└── requirements-train.txt       # heavier deps (torch, sentence-transformers)
```

## Quickstart

```bash
# 1. Install training deps
pip install -r requirements-train.txt

# 2. Copy data from mtg-oracle repo
cp ../mtg-oracle/data/oracle-slim.json.gz data/
cp ../mtg-oracle/data/mtg-axes.json data/

# 3. Generate soft labels (needs GPU or patience — ~10min on CPU)
python scripts/01_generate_labels.py

# 4. Train the student encoder (~2min on CPU)
python scripts/02_train_encoder.py

# 5. Validate
python scripts/03_validate.py
```

## Integration with mtg-oracle

Once trained, copy `models/encoder.pkl` to `mtg-oracle/lib/` and swap `score_deck()` in `lib/radar.py` for `embed_deck()` from this package. Same 105D output, same radar UI, better geometry.
