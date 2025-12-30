import os
import joblib
import re
import torch
import pandas as pd
from src.feature_extractor import token_to_features, add_neighboring_token_features
from src.cnn_model import LegalSBD_CNN
from src.crf_model import CONTEXT_WINDOW_SIZE, DELIMITERS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HYBRID_MODEL_PATH = os.path.join(BASE_DIR, "models", "crf_hybrid_model.joblib")
CNN_MODEL_PATH = os.path.join(BASE_DIR, "models", "cnn_model.pth")
TRAIN_CSV_PATH = os.path.join(BASE_DIR, "train_data.csv")

# Load hybrid CRF + CNN
hybrid_crf_model = joblib.load(HYBRID_MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_df = pd.read_csv(TRAIN_CSV_PATH)

all_chars = set()
for _, row in train_df.iterrows():
    all_chars.update(str(row["left_context"]))
    all_chars.update(str(row["delimiter"]))
    all_chars.update(str(row["right_context"]))
char_to_idx = {c: i + 2 for i, c in enumerate(sorted(all_chars))}
char_to_idx["<PAD>"] = 0
char_to_idx["<UNK>"] = 1
vocab_size = len(char_to_idx)

cnn_model = LegalSBD_CNN(vocab_size=vocab_size, embedding_dim=128, num_filters=6,
                         kernel_size=5, hidden_dim=250, dropout_prob=0.2).to(device)
cnn_model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=device))
cnn_model.eval()

def get_cnn_prediction_from_context(text, token_start_idx):
    if token_start_idx < 0 or token_start_idx >= len(text): return 0.0
    token = text[token_start_idx]
    if token not in DELIMITERS: return 0.0
    start_left = max(0, token_start_idx - CONTEXT_WINDOW_SIZE)
    right_end = token_start_idx + 1 + CONTEXT_WINDOW_SIZE
    sample_text = text[start_left:token_start_idx] + token + text[token_start_idx+1:right_end]
    max_len = (CONTEXT_WINDOW_SIZE*2) + 1
    padded = [char_to_idx.get(c, char_to_idx["<UNK>"]) for c in sample_text][:max_len] + [char_to_idx["<PAD>"]] * (max_len - len(sample_text))
    tensor = torch.tensor([padded], dtype=torch.long).to(device)
    with torch.no_grad():
        return cnn_model(tensor).item()

def segment_text(text: str):
    tokens_with_spans = [(m.group(0), m.start(), m.end()) for m in re.finditer(r"[\w'-]+|[.,!?;:()]|\S+", text)]
    if not tokens_with_spans: return []

    features_list = []
    cnn_probs = []
    for token, start, end in tokens_with_spans:
        features = token_to_features(token, text, start, end)
        if token in DELIMITERS:
            features["cnn_prob"] = round(get_cnn_prediction_from_context(text, start), 4)
            cnn_probs.append(features["cnn_prob"])
        else:
            cnn_probs.append(0.0)
        features_list.append(features)

    features_list = add_neighboring_token_features(features_list)
    try:
        labels = hybrid_crf_model.predict([features_list])[0]
    except:
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

    sentences = []
    current_start = 0
    for i, label in enumerate(labels):
        if label == "B":
            start_idx = tokens_with_spans[current_start][1]
            end_idx = tokens_with_spans[i][2]
            sentences.append(text[start_idx:end_idx].strip())
            current_start = i + 1

    # leftover
    if current_start < len(tokens_with_spans):
        start_idx = tokens_with_spans[current_start][1]
        sentences.append(text[start_idx:].strip())

    return sentences
