import argparse
import torch
import json
import numpy as np

from models.cnn_transformer_kan import CNNTransformerKAN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "cnn_transformer_final.pt"
LABEL_MAP_PATH = "label_map.json"

CNN_OUT_DIM = 64
D_MODEL = 128
NUM_HEADS = 4
NUM_LAYERS = 2
DROPOUT = 0.4


def load_model(num_classes):
    model = CNNTransformerKAN(
        num_classes=num_classes,
        cnn_out_dim=CNN_OUT_DIM,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def load_label_map():
    with open(LABEL_MAP_PATH, "r") as f:
        label_map = json.load(f)
    return {v: k for k, v in label_map.items()}


@torch.no_grad()
def run_inference(npy_path):
    x = np.load(npy_path).astype(np.float32)  # (9, 32, 8)
    x = torch.from_numpy(x).unsqueeze(0).to(DEVICE)  # (1, 9, 32, 8)

    idx_to_label = load_label_map()
    model = load_model(num_classes=len(idx_to_label))

    logits = model(x)
    probs = torch.softmax(logits, dim=1)

    pred_idx = probs.argmax(dim=1).item()
    confidence = probs.max(dim=1).values.item()

    return idx_to_label[pred_idx], confidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to a .npy sample file (shape: 9x32x8)")
    args = parser.parse_args()

    label, conf = run_inference(args.input)
    print(f"Predicted class : {label}")
    print(f"Confidence      : {conf:.4f}")
