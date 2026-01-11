# High-Frequency Trading Dynamics: A Multi-Channel Transformer Approach

This repository contains the official implementation of a **Multi-Channel Transformer** designed for Limit Order Book (LOB) prediction, featuring a **Liquid Neural Network (LNN)** extension.

## Overview
Predicting price movements in HFT requires handling high-dimensional data with extreme latency sensitivity. This project "elevates" traditional transformer models by integrating "Liquid" time-constant neurons to adapt to rapid market regime shifts.

## Project Structure
* `src/model.py`: Core architecture combining Multi-Channel Attention and the Liquid Layer extension.
* `src/train.py`: Training pipeline and evaluation logic.
* `requirements.txt`: Necessary Python libraries (PyTorch, Pandas, etc.).

## The Novel Extension
Unlike standard static models, our **Liquid Layer** uses differential equation-based state updates:
- **Continuous Adaptation:** Adjusts to irregular time intervals in tick data.
- **Regime Robustness:** Dynamically tunes the 'tau' parameter to handle volatility.

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
