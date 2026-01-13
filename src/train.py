import torch
import torch.nn as nn
import torch.optim as optim
from model import LiquidTransformer # Import the class we just made

# 1. Setup Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LiquidTransformer().to(device)
criterion = nn.CrossEntropyLoss() # Standard for 3-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def load_fi2010_data(file_path):
    # Load the text file (space-separated)
    # FI-2010 .txt files usually have 149 columns. 
    # Columns 1-40 are features, 149 is the k=100 label.
    data = np.loadtxt(file_path) 
    
    # Transpose if the data is saved as (Features, Time) instead of (Time, Features)
    if data.shape[0] < data.shape[1]:
        data = data.T
        
    features = torch.tensor(data[:, :40], dtype=torch.float32)
    # Column index 148 is typically the label for k=100
    labels = torch.tensor(data[:, 148] - 1, dtype=torch.long) # -1 to make labels 0, 1, 2
    
    # Create the 100-tick time window
    features = features.unfold(0, 100, 1).permute(0, 1, 2)
    labels = labels[99:] 
    
    return DataLoader(TensorDataset(features, labels), batch_size=32, shuffle=True)
    
def train_model(train_loader, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            # CRITICAL: Push data to GPU accelerator
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs) # Passes through the Liquid evolution loop
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        print(f"Epoch {epoch+1} | Avg Loss: {running_loss/len(train_loader):.4f}")

from sklearn.metrics import accuracy_score, precision_score, f1_score

def evaluate_and_log(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics using 'macro' average as done in the FI-2010 benchmarks
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    print(f"\n--- FINAL RESULTS FOR TABLE ---")
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {prec*100:.2f}%")
    print(f"F1-Score:  {f1*100:.2f}%")
    print(f"--------------------------------\n")

if __name__ == "__main__":
    # Update this path to where your FI-2010 dataset is in Kaggle
    DATA_PATH = '/kaggle/input/limit-orderbook-data/Train_Dst_NoAuction_DecPre_CF_7.txt' 
    
    # This prepares the data for the Liquid Transformer
    print("Loading FI-2010 Dataset...")
    train_loader = load_fi2010_data(DATA_PATH)
    
    # This starts the actual learning process on the GPU
    print("Starting Training...")
    train_model(train_loader, epochs=10)
    
