# High-Frequency Trading Dynamics: A Multi-Channel Transformer Approach

[![Paper](https://img.shields.io/badge/Whitepaper-PDF-blue)](whitepaper/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-FI--2010-green)](https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649)

A Multi-Channel Transformer architecture for Limit Order Book (LOB) 
mid-price forecasting in high-frequency trading environments, featuring 
a **Liquid Neural Network (LNN)** integration head for improved temporal 
robustness across prediction horizons.

---

## Results

Evaluated across 5 subsets (CF_1, CF_3, CF_5, CF_7, CF_9) of the 
FI-2010 NoAuction Z-Score dataset at horizon k=100:

| Model | Accuracy | Precision | F1-Score |
|---|---|---|---|
| MLP Baseline | 41.20% | 42.10% | 0.4012 |
| CNN-LOB | 54.12% | 55.08% | 0.5390 |
| **Liquid Transformer (Ours)** | **54.80% ± 2.80%** | **53.33% ± 4.05%** | **0.5327 ± 0.0419** |
| TransLOB (State-of-the-Art) | 62.10% | 61.40% | 0.6120 |

**Inference latency:** 0.6711 ms per tick (~1,490 ticks/s) on NVIDIA T4 GPU.

---

## Architecture

The model processes LOB data as a 3D tensor X ∈ R^(H×W×C) where:
- **H** = price level depth (top 10 bid/ask levels)
- **W** = time window (100 ticks)
- **C** = channels (price, volume)

Four stages:
1. **Linear Patch Projection** — spatial-temporal patch embedding
2. **Dual Positional Encoding** — temporal + spatial index injection
3. **Dual-Attention Encoder** — spatial attention (price levels) + temporal attention (tick sequence)
4. **Liquid Integration Head** — LTC layer replacing standard pooling, governed by:

$$\frac{dx}{dt} = -[A + S(t)]x(t) + S(t)I(t)$$

---

## Project Structure

```
LOB-Transformer-Quant/
├── src/
│   ├── model.py          # LiquidTransformer architecture
│   └── train.py          # Training loop, evaluation, latency benchmark
├── notebooks/            # Kaggle training notebooks
├── data/                 # README with FI-2010 download instructions
├── whitepaper/           # PDF of the research paper
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/nyda1908/LOB-Transformer-Quant.git
cd LOB-Transformer-Quant
pip install -r requirements.txt
```

**Dataset:** Download FI-2010 from the 
[official source](https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649) 
and place the NoAuction_Zscore files under `data/`.

---

## Training

```bash
python src/train.py
```

To train across all subsets:
```python
# In train.py, set:
subsets = [1, 3, 5, 7, 9]
```

---

## Key Findings

- **Weighted cross-entropy loss** applied to handle class imbalance 
  across Up/Down/Stable labels
- **Per-subset variance** (F1: 0.4687–0.5836) reflects genuine 
  stock-specific microstructure differences
- **Slower F1 decay** across horizons vs CNN-LOB baseline, validating 
  temporal robustness of liquid neurons

---

## References

1. Tsantekidis et al., "Forecasting Stock Prices from the LOB using CNNs," IEEE CBI, 2017
2. Ntakaris et al., "Benchmark Dataset for Mid-Price Forecasting," Journal of Forecasting, 2018
3. Hasani et al., "Liquid Time-Constant Networks," AAAI, 2021
4. Vaswani et al., "Attention is All You Need," NeurIPS, 2017

---

## Author

**Nidhi Maheshwari**  
Department of Mathematics, IIT Kharagpur  
nidhi.m.24@kgpian.iitkgp.ac.in
