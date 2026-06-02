import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os #added for path handling
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, f1_score

#REPRODUCIBILITY SEED 
torch.manual_seed(42)
np.random.seed(42)

#1.DATA LOADER
def load_fi2010_data(file_path):
    print(f"Reading: {file_path}")
    data = np.loadtxt(file_path).T 
    features = torch.tensor(data[:, :40], dtype=torch.float32)
    labels = torch.tensor(data[:, 148] - 1, dtype=torch.long) 
    features = features.unfold(0, 100, 1).transpose(1, 2)
    labels = labels[99:] 
    return DataLoader(TensorDataset(features, labels), batch_size=32, shuffle=True)

#2.MODEL DEFINITION
class LiquidTransformer(nn.Module):
    def __init__(self, input_dim=40, d_model=64, nhead=4, num_layers=2):
        super(LiquidTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 3)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        return self.fc(x[:, -1, :])

#3.METRICS HELPER
def get_all_metrics(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)           
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return acc, prec, f1

# weights for the loss function
def compute_class_weights(data_path, device):
    data = np.loadtxt(data_path)
    labels = data[-1, :].astype(int) - 1  # convert 1,2,3 → 0,1,2
    class_counts = np.bincount(labels)
    total = len(labels)
    weights = total / (len(class_counts) * class_counts)    
    print(f"Class weights: {weights}")
    return torch.FloatTensor(weights).to(device)

#4.MAIN TRAINING SESSION
def train_full_session(epochs=20):
    #KAGGLE PATHS
    TRAIN_PATH = '/kaggle/input/fi2010/NoAuction_Zscore_Training/Train_Dst_NoAuction_ZScore_CF_7.txt'
    TEST_PATH = '/kaggle/input/fi2010/NoAuction_Zscore_Testing/Test_Dst_NoAuction_ZScore_CF_7.txt'
    #ABSOLUTE SAVE PATH
    SAVE_PATH = '/kaggle/working/best_model.pt'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LiquidTransformer().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001) 
    weights = compute_class_weights(TRAIN_PATH, device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    
    train_loader = load_fi2010_data(TRAIN_PATH)
    test_loader = load_fi2010_data(TEST_PATH)
    
    best_f1 = 0.0
    print(f"\n{'Epoch':<6} | {'Loss':<8} | {'Acc':<8} | {'Prec':<8} | {'F1':<8}")
    print("-" * 55)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        acc, prec, f1 = get_all_metrics(model, test_loader, device)
        
        print(f"{epoch+1:02d}    | {avg_loss:.4f} | {acc:.4f} | {prec:.4f} | {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            #Using absolute path for reliable Kaggle output
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"New Best Model Saved (F1: {f1:.4f}) at {SAVE_PATH}")

    #Final Directory Verification
    print("\nTraining Complete.")
    print(f"Files in Output Directory: {os.listdir('/kaggle/working')}")

def train_and_evaluate_all_subsets(epochs=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subsets = [1, 3, 5, 7, 9]
    results = []

    for cf in subsets:
        print(f"\n{'='*55}")
        print(f"Training on CF_{cf}")
        print(f"{'='*55}")

        TRAIN_PATH = f'/kaggle/input/fi2010/NoAuction_Zscore_Training/Train_Dst_NoAuction_ZScore_CF_{cf}.txt'
        TEST_PATH = f'/kaggle/input/fi2010/NoAuction_Zscore_Testing/Test_Dst_NoAuction_ZScore_CF_{cf}.txt'
  
        model = LiquidTransformer().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        weights = compute_class_weights(TRAIN_PATH, device)
        criterion = nn.CrossEntropyLoss(weight=weights)

        train_loader = load_fi2010_data(TRAIN_PATH)
        test_loader = load_fi2010_data(TEST_PATH)

        best_f1, best_acc, best_prec = 0.0, 0.0, 0.0

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(inputs), labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            acc, prec, f1 = get_all_metrics(model, test_loader, device)
            if f1 > best_f1:
                best_f1, best_acc, best_prec = f1, acc, prec
                print(f"  Epoch {epoch+1:02d} — New Best F1: {f1:.4f}")

        results.append({
            'subset': f'CF_{cf}',
            'accuracy': best_acc,
            'precision': best_prec,
            'f1': best_f1
        })
        print(f"CF_{cf} Final — Acc: {best_acc:.4f} | Prec: {best_prec:.4f} | F1: {best_f1:.4f}")

    # Summary
    print(f"\n{'='*55}")
    print("SUMMARY ACROSS SUBSETS")
    print(f"{'='*55}")
    accs = [r['accuracy'] for r in results]
    precs = [r['precision'] for r in results]
    f1s = [r['f1'] for r in results]
    for r in results:
        print(f"{r['subset']} | Acc: {r['accuracy']:.4f} | Prec: {r['precision']:.4f} | F1: {r['f1']:.4f}")
        print(f"\nMean | Acc: {np.mean(accs):.4f} | Prec: {np.mean(precs):.4f} | F1: {np.mean(f1s):.4f}")
        print(f"Std  | Acc: {np.std(accs):.4f} | Prec: {np.std(precs):.4f} | F1: {np.std(f1s):.4f}")

train_and_evaluate_all_subsets(20)
