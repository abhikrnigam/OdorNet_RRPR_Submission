# OdorNet: E-Nose Odor Classification with CNN-Transformer-KAN

This repository contains the official implementation of **OdorNet**, an electronic nose (e-nose) odor classification system proposed in our ICPR 2026 paper. The model combines a 1D CNN encoder, a Transformer encoder, and a Kolmogorov-Arnold Network (KAN) classification head to classify 50 distinct odor categories from 8-channel gas sensor data.

---

## Model Architecture

The proposed **CNN-Transformer-KAN** model processes e-nose sensor readings as a sequence of temporal patches:

```
Input (B, 9 patches, 32 time steps, 8 sensors)
    ↓
1D CNN Encoder     — extracts local temporal features per patch
    ↓
Transformer Encoder — models cross-patch dependencies (2 layers, 4 heads)
    ↓
Global Pooling     — aggregates patch representations
    ↓
KAN Head           — learned 1D basis functions for classification
    ↓
Logits (50 classes)
```

| Component | Details |
|---|---|
| CNN | Conv1d(8→32→64), GELU, temporal mean pooling |
| Transformer | d_model=128, 4 heads, 2 layers, dropout=0.4 |
| KAN Head | 128 independent learned MLPs summed per output class |
| Optimizer | AdamW (lr=1e-4, weight_decay=1e-4) |
| Loss | CrossEntropyLoss |
| Epochs | 500 |

---

## Dataset

The **OdorNet Dataset** contains sensor readings from 50 odor categories across 5 augmentation types (original, jitter, scaling, time warp, sensor drift), yielding 1,250 samples total.

**Download:** [OdorNet Dataset on Google Drive](https://drive.google.com/drive/folders/1c4Url2e1Mh6IKcmydLvuplJjsLhlX1ZC?usp=sharing)

After downloading, place the dataset folder as:
```
Enose/
└── processed_data/
    ├── orig/
    ├── jitter/
    ├── scaling/
    ├── time_warp/
    └── drift/
```

Each subdirectory contains 50 class folders, each with 5 `.npy` files of shape `(9, 32, 8)` — 9 patches × 32 time steps × 8 sensor channels.

---

## Requirements

- Python >= 3.10
- PyTorch >= 2.0
- CUDA (optional, CPU also supported)

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Setup & Run

### 1. Clone the repository

```bash
git clone https://github.com/abhikrnigam/Enose.git
cd Enose
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download from the [Google Drive link](https://drive.google.com/drive/folders/1c4Url2e1Mh6IKcmydLvuplJjsLhlX1ZC?usp=sharing) and place it at `processed_data/` inside the repo root.

### 5. Train the model

```bash
# With Weights & Biases logging (requires wandb account)
python train_cnn_transformer_kan.py

# Without Weights & Biases (offline mode, no account needed)
python train_cnn_transformer_kan.py --no-wandb
```

Training will produce:
- `cnn_transformer_final.pt` — trained model weights
- `label_map.json` — class index mapping

### 6. Run inference on a single sample

```bash
python inference.py --input path/to/sample.npy
```

---

## Repository Structure

```
Enose/
├── models/
│   ├── cnn_transformer_kan.py   # Proposed SOTA model
│   ├── kan.py                   # KAN layer implementation
│   ├── cnn_transformer.py       # Baseline: CNN + Transformer
│   ├── transformer.py           # Baseline: Transformer only
│   ├── lstm.py                  # Baseline: LSTM
│   ├── rnn.py                   # Baseline: RNN
│   └── mlp.py                   # Baseline: MLP
├── utils/
│   ├── normalisation.py
│   ├── segmentation.py
│   ├── jitter.py
│   ├── scaling.py
│   ├── time_warping.py
│   ├── sensor_drift.py
│   └── outlier_removal.py
├── train_cnn_transformer_kan.py  # Training script (proposed model)
├── train_cnn_transformer.py      # Training script (baseline)
├── train_lstm.py
├── train_rnn.py
├── train_transformer.py
├── train_mlp.py
├── dataloader_updated.py         # DataLoader factory
├── dataset.py                    # ENoseDataset class
├── inference.py                  # Single-sample inference
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Baseline Comparisons

The following baseline models are included for comparison:

| Model | Script | Top-1 Acc |
|---|---|---|
| CNN + Transformer + KAN (proposed) | `train_cnn_transformer_kan.py` | 75.3% |
| CNN + Transformer (ablation) | `train_cnn_transformer.py` | 56.2% |
| Encoder-Only Transformer | `train_transformer.py` | 45.7% |
| MLP | `train_mlp.py` | 33.3% |
| LSTM | `train_lstm.py` | 32.8% |
| RNN | `train_rnn.py` | 30.9% |

All baselines use the same dataset split and dataloader. For full architecture details, layer configurations, and hyperparameters of every model, see [MODELS.md](MODELS.md).

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
