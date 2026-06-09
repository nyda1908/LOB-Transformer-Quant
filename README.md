# High-Frequency Trading Dynamics: A Multi-Channel Transformer Approach

[![Paper](https://img.shields.io/badge/Whitepaper-PDF-blue)](whitepaper/LOB_Transformer_Whitepaper.pdf)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-FI--2010-green)](https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649)

A Multi-Channel Transformer architecture for Limit Order Book (LOB) 
mid-price forecasting in high-frequency trading environments, featuring 
a **Liquid Neural Network (LNN)** integration head for improved temporal 
robustness across prediction horizons.

---

## Results

Evaluated across 3 subsets (CF_1, CF_5, CF_9) of the FI-2010 NoAuction Z-Score dataset at horizon k=100:

| Model | Accuracy | Precision | F1-Score |
|---|---|---|---|
| MLP Baseline | 41.20% | 42.10% | 0.4012 |
| CNN-LOB | 54.12% | 55.08% | 0.5390 |
| **Liquid Transformer (Ours)** | **58.64% ± 6.21%** | **58.26% ± 6.93%** | **0.5864 ± 0.0636** |
| TransLOB (State-of-the-Art) | 62.10% | 61.40% | 0.6120 |

**Ablation:** The LTC head improves mean F1 by +0.060 over a plain Transformer baseline (0.5864 vs 0.5259).

Inference latency: 3.56 ms per tick (~280 ticks/s) after TorchScript JIT compilation on NVIDIA T4 GPU. The LTC head introduces a sequential dependency, representing an accuracy-latency tradeoff (see whitepaper).

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
subsets = [1, 5, 9]
```

---

## Key Findings

- **Ablation:** The LTC head improves mean F1 by +0.060 over a plain Transformer baseline (0.5864 vs 0.5259), confirming it is the primary driver of performance gains.
- **Per-subset variance** (F1: 0.4980–0.6529) reflects genuine stock-specific microstructure differences.
- **Weighted cross-entropy loss** applied to handle class imbalance across Up/Down/Stable labels.
- **Accuracy-latency tradeoff:** The LTC head improves accuracy but adds a sequential dependency (3.56 ms/tick after JIT). The advantage is largest at short horizons (+0.143 at k=10) and narrows at k=100 (+0.054).
  
---

## References

1. Ntakaris et al., "Benchmark Dataset for Mid-Price Forecasting of Limit Order Book Data," Journal of Forecasting, 2018
2. Tsantekidis et al., "Forecasting Stock Prices from the Limit Order Book using CNNs," IEEE CBI, 2017
3. Wallbridge, J., "Transformers for Limit Order Books," arXiv:2003.00130, 2020
4. Hasani et al., "Liquid Time-Constant Networks," AAAI, 2021
5. Vaswani et al., "Attention is All You Need," NeurIPS, 2017
---

## Author

**Nidhi Maheshwari**  
Department of Mathematics, IIT Kharagpur  
nidhi.m.24@kgpian.iitkgp.ac.in
