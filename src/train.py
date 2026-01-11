import torch
import torch.nn as nn
import torch.optim as optim
from model import LiquidTransformer # Import the class we just made

# 1. Setup Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LiquidTransformer().to(device)
criterion = nn.CrossEntropyLoss() # Standard for 3-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 2. Mock Data for Testing
# Represents (Batch Size=32, Time Steps=100, LOB Features=40)
dummy_input = torch.randn(32, 100, 40).to(device)
dummy_labels = torch.randint(0, 3, (32,)).to(device) # Up, Down, Neutral

# 3. The Training Loop
def train_one_epoch():
    model.train()
    optimizer.zero_grad()
    
    # Forward Pass
    outputs = model(dummy_input)
    loss = criterion(outputs, dummy_labels)
    
    # Backward Pass (The "Learning" part)
    loss.backward()
    optimizer.step()
    
    print(f"Epoch Complete. Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train_one_epoch()
