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

The **OdorNet Dataset** contains sensor readings from 50 odor categories, with 25 recordings per class across 50 classes — 1,250 samples total. Each sample is a `.npy` file of shape `(9, 32, 8)` — 9 patches × 32 time steps × 8 sensor channels.

**Download:** [OdorNet Dataset on Google Drive](https://drive.google.com/drive/folders/11X91-wKewPPFiLBCo19TKjICJOGwFzmn?usp=sharing)

After downloading, place the dataset folder at the repo root:
```
OdorNet_RRPR_Submission/
└── OdorNet_Dataset/
    ├── ajwain/
    │   ├── ajwain_cycle_0.npy
    │   ├── ajwain_cycle_1.npy
    │   └── ... (25 files)
    ├── almonds/
    └── ... (50 classes total)
```

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
git clone https://github.com/abhikrnigam/OdorNet_RRPR_Submission.git
cd OdorNet_RRPR_Submission
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

Download from the [Google Drive link](https://drive.google.com/drive/folders/11X91-wKewPPFiLBCo19TKjICJOGwFzmn?usp=sharing) and place the `OdorNet_Dataset/` folder at the repo root as shown above.

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
OdorNet_RRPR_Submission/
├── OdorNet_Dataset/              # Dataset (download from Google Drive)
│   ├── ajwain/
│   ├── almonds/
│   └── ... (50 classes)
├── models/
│   ├── cnn_transformer_kan.py   # Proposed SOTA model
│   └── kan.py                   # KAN layer implementation
├── train_cnn_transformer_kan.py  # Training script
├── dataloader_updated.py         # DataLoader factory
├── dataset.py                    # ENoseDataset class
├── inference.py                  # Single-sample inference
├── cnn_transformer_final.pt      # Trained model weights
├── label_map.json                # Class index mapping
├── requirements.txt
├── MODELS.md                     # Architecture & hyperparameter details for all models
├── LICENSE
└── README.md
```

---

## Baseline Comparisons

The following baseline models were evaluated in the paper:

| Model | Top-1 Acc |
|---|---|
| CNN + Transformer + KAN (proposed) | 75.3% |
| CNN + Transformer (ablation) | 56.2% |
| Encoder-Only Transformer | 45.7% |
| MLP | 33.3% |
| LSTM | 32.8% |
| RNN | 30.9% |

For full architecture details, layer configurations, and hyperparameters of every model, see [MODELS.md](MODELS.md).

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
