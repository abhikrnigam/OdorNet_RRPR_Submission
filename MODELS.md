# OdorNet — Model Configurations & Hyperparameters

This document covers all models evaluated in the paper: the proposed OdorNet (CNN-Transformer-KAN), the ablation variant (CNN-Transformer), and the four baselines (MLP, RNN, LSTM, Encoder-Only Transformer).

All models are trained and evaluated on the **OdorDB dataset** — 50 odor classes, input shape `(B, 9, 32, 8)` (batch × patches × time steps × sensor channels).

---

## 1. OdorNet — CNN + Transformer + KAN *(Proposed)*

**Paper result: Top-1 75.3% | Top-5 92.2%**

### Architecture

| Stage | Component | Details |
|---|---|---|
| 1 | Reshape | `(B, 9, 32, 8)` → `(B×9, 8, 32)` |
| 2 | Conv1d #1 | `in=8, out=32, kernel=3, padding=1` + GELU |
| 3 | Conv1d #2 | `in=32, out=64, kernel=3, padding=1` + GELU |
| 4 | Temporal Pooling | Mean over time axis → `(B×9, 64)` |
| 5 | Linear Projection | `Linear(64 → 128)` |
| 6 | Reshape | `(B×9, 128)` → `(B, 9, 128)` |
| 7 | Transformer Encoder | 2 layers, d_model=128, 4 heads, dropout=0.4 |
| 8 | LayerNorm | `LayerNorm(128)` |
| 9 | Global Pooling | Mean over patches → `(B, 128)` |
| 10 | KAN Head | `KANLayer(in=128, out=50, hidden=32)` |

### KAN Layer
Each of the 128 input dimensions has its own independent MLP: `Linear(1→32) → GELU → Linear(32→num_classes)`. Outputs are summed across all 128 dimensions to produce class logits.

### Hyperparameters

| Parameter | Value |
|---|---|
| `cnn_out_dim` | 64 |
| `d_model` | 128 |
| `num_heads` | 4 |
| `num_layers` | 2 |
| `dropout` | 0.4 |
| `kan_hidden_dim` | 32 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Batch size | 8 |
| Epochs | 500 |
| Loss | CrossEntropyLoss |

---

## 2. CNN + Transformer *(Ablation — no KAN)*

**Paper result: Top-1 56.2%**

Same as OdorNet but replaces the KAN head with a standard MLP classifier. Uses a `PatchCNNEncoder` with learnable positional embeddings added before the Transformer.

### Architecture

| Stage | Component | Details |
|---|---|---|
| 1 | CNN Encoder | Conv1d blocks → `(B×9, 64)` |
| 2 | Linear + LayerNorm | `Linear(64 → 128)` + `LayerNorm(128)` |
| 3 | Positional Embedding | Learnable `(1, 9, 128)` |
| 4 | Transformer Encoder | 2 layers, d_model=128, 4 heads, dropout=0.4 |
| 5 | Mean Pooling | Over patches → `(B, 128)` |
| 6 | MLP Classifier | `Linear(128→128) → ReLU → Dropout → Linear(128→50)` |

### Hyperparameters

| Parameter | Value |
|---|---|
| `cnn_out_dim` | 64 |
| `d_model` | 128 |
| `num_heads` | 4 |
| `num_layers` | 2 |
| `dropout` | 0.4 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Batch size | 8 |
| Epochs | 500 |
| Loss | CrossEntropyLoss |

---

## 3. Encoder-Only Transformer *(Baseline)*

**Paper result: Top-1 45.7%**

### Architecture

| Stage | Component | Details |
|---|---|---|
| 1 | Flatten patches | `(B, 9, 32, 8)` → `(B, 9, 256)` |
| 2 | Patch Embedding | `Linear(256 → 256)` |
| 3 | Positional Embedding | Learnable `(1, 9, 256)` |
| 4 | Transformer Encoder | 2 layers, d_model=256, 4 heads, FFN dim=1024, GELU, dropout=0.4 |
| 5 | Mean Pooling | Over tokens → `(B, 256)` |
| 6 | MLP Classifier | `Linear(256→256) → ReLU → Dropout → Linear(256→50)` |

### Hyperparameters

| Parameter | Value |
|---|---|
| `input_dim` | 256 (32×8 flattened) |
| `d_model` | 256 |
| `num_heads` | 4 |
| `num_layers` | 2 |
| `dropout` | 0.4 |
| `max_len` | 9 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Batch size | 8 |
| Epochs | 500 |
| Loss | CrossEntropyLoss |

---

## 4. LSTM *(Baseline)*

**Paper result: Top-1 32.8%**

### Architecture

| Stage | Component | Details |
|---|---|---|
| 1 | Flatten patches | `(B, 9, 32, 8)` → `(B, 9, 256)` |
| 2 | LSTM | `input=256, hidden=128, layers=1` |
| 3 | Last timestep | `out[:, -1, :]` → `(B, 128)` |
| 4 | Dropout | `p=0.4` |
| 5 | MLP Classifier | `Linear(128→128) → ReLU → Dropout(0.4) → Linear(128→50)` |

### Hyperparameters

| Parameter | Value |
|---|---|
| `input_dim` | 256 (32×8 flattened) |
| `hidden_dim` | 128 |
| `num_layers` | 1 |
| `dropout` | 0.4 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Batch size | 8 |
| Epochs | 500 |
| Loss | CrossEntropyLoss |

---

## 5. RNN *(Baseline)*

**Paper result: Top-1 30.9%**

### Architecture

| Stage | Component | Details |
|---|---|---|
| 1 | Flatten patches | `(B, 9, 32, 8)` → `(B, 9, 256)` |
| 2 | RNN | `input=256, hidden=256, layers=1`, tanh activation |
| 3 | Last timestep | `out[:, -1, :]` → `(B, 256)` |
| 4 | MLP Classifier | `Linear(256→256) → ReLU → Dropout(0.3) → Linear(256→50)` |

### Hyperparameters

| Parameter | Value |
|---|---|
| `input_dim` | 256 (32×8 flattened) |
| `hidden_dim` | 256 |
| `num_layers` | 1 |
| `dropout` | 0.3 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Batch size | 8 |
| Epochs | 500 |
| Loss | CrossEntropyLoss |

---

## 6. MLP *(Baseline)*

**Paper result: Top-1 33.3%**

### Architecture

| Stage | Component | Details |
|---|---|---|
| 1 | Flatten | `(B, 9, 32, 8)` → `(B, 2304)` |
| 2 | FC Layer 1 | `Linear(2304→1024) → ReLU → BN → Dropout(0.3)` |
| 3 | FC Layer 2 | `Linear(1024→256) → ReLU → BN → Dropout(0.3)` |
| 4 | Output | `Linear(256→50)` |

### Hyperparameters

| Parameter | Value |
|---|---|
| `input_dim` | 2304 (9×32×8 flattened) |
| Hidden dims | 1024 → 256 |
| `dropout` | 0.3 |
| Batch Norm | After each hidden layer |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Batch size | 8 |
| Epochs | 500 |
| Loss | CrossEntropyLoss |

---

## Summary Table

| Model | Top-1 Acc (%) | Key Design Choice |
|---|---|---|
| **OdorNet (CNN + Transformer + KAN)** | **75.3** | KAN head for nonlinear decision boundaries |
| CNN + Transformer (ablation) | 56.2 | MLP head instead of KAN |
| Encoder-Only Transformer | 45.7 | No CNN, direct patch embedding |
| MLP | 33.3 | Fully flattened input, no temporal modeling |
| LSTM | 32.8 | Sequential modeling, hidden=128 |
| RNN | 30.9 | Sequential modeling, hidden=256, tanh |
