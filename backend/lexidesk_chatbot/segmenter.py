# backend/lexidesk_chatbot/segmenter.py

import re
import torch
import joblib
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# Absolute path resolution (SAFE)
# --------------------------------------------------

# backend/lexidesk_chatbot/segmenter.py → backend/
BACKEND_ROOT = Path(__file__).resolve().parents[1]

MODELS_DIR = BACKEND_ROOT / "models"
SRC_DIR = BACKEND_ROOT / "src"

HYBRID_MODEL_PATH = MODELS_DIR / "crf_hybrid_model.joblib"
CNN_MODEL_PATH = MODELS_DIR / "cnn_model.pth"
TRAIN_CSV_PATH = BACKEND_ROOT / "train_data.csv"

# --------------------------------------------------
# Imports from backend/src
# --------------------------------------------------

import sys
sys.path.append(str(SRC_DIR))

from feature_extractor import token_to_features, add_neighboring_token_features
from cnn_model import LegalSBD_CNN
from crf_model import CONTEXT_WINDOW_SIZE, DELIMITERS

# --------------------------------------------------
# Lazy-loaded globals
# --------------------------------------------------

_hybrid_crf_model = None
_cnn_model = None
_char_to_idx = None

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# Model loader
# --------------------------------------------------

def _load_models():
    global _hybrid_crf_model, _cnn_model, _char_to_idx

    if _hybrid_crf_model is not None:
        return

    # ---- CRF ----
    if HYBRID_MODEL_PATH.exists():
        _hybrid_crf_model = joblib.load(HYBRID_MODEL_PATH)
    else:
        print("[WARN] CRF model not found, using fallback")
        _hybrid_crf_model = None

    # ---- CNN ----
    if CNN_MODEL_PATH.exists() and TRAIN_CSV_PATH.exists():
        train_df = pd.read_csv(TRAIN_CSV_PATH)

        all_chars = set()
        for _, row in train_df.iterrows():
            all_chars.update(str(row["left_context"]))
            all_chars.update(str(row["delimiter"]))
            all_chars.update(str(row["right_context"]))

        _char_to_idx = {c: i + 2 for i, c in enumerate(sorted(all_chars))}
        _char_to_idx["<PAD>"] = 0
        _char_to_idx["<UNK>"] = 1

        _cnn_model = LegalSBD_CNN(
            vocab_size=len(_char_to_idx),
            embedding_dim=128,
            num_filters=6,
            kernel_size=5,
            hidden_dim=250,
            dropout_prob=0.2,
        ).to(_device)

        _cnn_model.load_state_dict(
            torch.load(CNN_MODEL_PATH, map_location=_device)
        )
        _cnn_model.eval()
    else:
        _cnn_model = None
        _char_to_idx = None

# --------------------------------------------------
# CNN helper
# --------------------------------------------------

def _cnn_prob(text, idx):
    if _cnn_model is None or _char_to_idx is None:
        return 0.0

    token = text[idx]
    if token not in DELIMITERS:
        return 0.0

    start = max(0, idx - CONTEXT_WINDOW_SIZE)
    end = idx + 1 + CONTEXT_WINDOW_SIZE
    sample = text[start:end]

    max_len = (CONTEXT_WINDOW_SIZE * 2) + 1
    padded = [
        _char_to_idx.get(c, _char_to_idx["<UNK>"])
        for c in sample[:max_len]
    ] + [_char_to_idx["<PAD>"]] * (max_len - len(sample))

    tensor = torch.tensor([padded], dtype=torch.long).to(_device)

    with torch.no_grad():
        return _cnn_model(tensor).item()

# --------------------------------------------------
# Public API
# --------------------------------------------------

def segment_text(text: str):
    _load_models()

    tokens = [
        (m.group(0), m.start(), m.end())
        for m in re.finditer(r"[\w'-]+|[.,!?;:()]|\S+", text)
    ]

    if not tokens:
        return []

    features = []
    for tok, start, end in tokens:
        f = token_to_features(tok, text, start, end)
        f["cnn_prob"] = _cnn_prob(text, start)
        features.append(f)

    features = add_neighboring_token_features(features)

    labels = None
    if _hybrid_crf_model:
        try:
            labels = _hybrid_crf_model.predict([features])[0]
        except Exception:
            labels = None

    if labels is None:
        return [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", text)
            if s.strip()
        ]

    sentences = []
    start_i = 0
    for i, lbl in enumerate(labels):
        if lbl == "B":
            s = tokens[start_i][1]
            e = tokens[i][2]
            sentences.append(text[s:e].strip())
            start_i = i + 1

    if start_i < len(tokens):
        sentences.append(text[tokens[start_i][1]:].strip())

    return sentences
