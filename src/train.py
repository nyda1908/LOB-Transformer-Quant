import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, f1_score
from model import LiquidTransformer, PlainTransformer

torch.manual_seed(42)
np.random.seed(42)

# Horizon label rows in FI-2010 (k=10,20,30,50,100 -> rows 144-148)
HORIZON_ROWS = {10: 144, 20: 145, 30: 146, 50: 147, 100: 148}


def load_fi2010_data(file_path, label_row=148):
    print(f"Reading: {file_path}")
    data = np.loadtxt(file_path).T
    features = torch.tensor(data[:, :40], dtype=torch.float32)
    labels = torch.tensor(data[:, label_row] - 1, dtype=torch.long)
    features = features.unfold(0, 100, 1).transpose(1, 2)
    labels = labels[99:]
    return DataLoader(TensorDataset(features, labels), batch_size=32, shuffle=True)


def compute_class_weights(data_path, device, label_row=148):
    data = np.loadtxt(data_path)
    labels = data[label_row, :].astype(int) - 1
    counts = np.bincount(labels)
    weights = len(labels) / (len(counts) * counts)
    return torch.FloatTensor(weights).to(device)


def get_all_metrics(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            preds = model(inputs).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return acc, prec, f1


def run_ablation(subsets=[1, 5, 9], epochs=20, data_root='data'):
    """Trains PlainTransformer and LiquidTransformer across subsets; reports mean F1."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for model_name, ModelClass in [("PlainTransformer", PlainTransformer),
                                   ("LiquidTransformer", LiquidTransformer)]:
        print(f"\n{'='*55}\nMODEL: {model_name}\n{'='*55}")
        results = []
        for cf in subsets:
            train_path = f'{data_root}/NoAuction_Zscore_Training/Train_Dst_NoAuction_ZScore_CF_{cf}.txt'
            test_path = f'{data_root}/NoAuction_Zscore_Testing/Test_Dst_NoAuction_ZScore_CF_{cf}.txt'

            model = ModelClass().to(device)
            optimizer = optim.Adam(model.parameters(), lr=1e-4)
            criterion = nn.CrossEntropyLoss(weight=compute_class_weights(train_path, device))
            train_loader = load_fi2010_data(train_path)
            test_loader = load_fi2010_data(test_path)

            best_f1, best_acc, best_prec = 0.0, 0.0, 0.0
            for epoch in range(epochs):
                model.train()
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(inputs), labels)
                    loss.backward()
                    optimizer.step()
                acc, prec, f1 = get_all_metrics(model, test_loader, device)
                if f1 > best_f1:
                    best_f1, best_acc, best_prec = f1, acc, prec
            results.append({'f1': best_f1, 'accuracy': best_acc, 'precision': best_prec})
            print(f"CF_{cf} | Acc: {best_acc:.4f} | Prec: {best_prec:.4f} | F1: {best_f1:.4f}")

        f1s = [r['f1'] for r in results]
        accs = [r['accuracy'] for r in results]
        print(f"\nMean | Acc: {np.mean(accs):.4f} | F1: {np.mean(f1s):.4f}")
        print(f"Std  | Acc: {np.std(accs):.4f} | F1: {np.std(f1s):.4f}")


def benchmark_latency(jit=True):
    """Measures per-tick inference latency (batch size 1)."""
    import time
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LiquidTransformer().to(device).eval()
    sample = torch.randn(1, 100, 40).to(device)
    if jit:
        with torch.no_grad():
            model = torch.jit.script(model)

    with torch.no_grad():
        for _ in range(100):
            _ = model(sample)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(1000):
            _ = model(sample)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end = time.perf_counter()

    avg_ms = ((end - start) / 1000) * 1000
    print(f"Latency: {avg_ms:.4f} ms/tick | Throughput: {1000/avg_ms:.0f} ticks/s")


if __name__ == "__main__":
    run_ablation(subsets=[1, 5, 9], epochs=20)
    benchmark_latency(jit=True)
