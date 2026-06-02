import torch
import torch.nn as nn
import time
import sys

#1.ARCHITECTURE DEFINITION
class LiquidTransformer(nn.Module):
    def __init__(self):
        super(LiquidTransformer, self).__init__()
        self.embedding = nn.Linear(40, 64)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=64, nhead=4, batch_first=True), 
            num_layers=2
        )
        self.fc = nn.Linear(64, 3)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        return self.fc(x[:, -1, :])

#2.THE BENCHMARK FUNCTION
def run_final_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Targeting Device: {device}", flush=True)
    
    #initialize and load
    model = LiquidTransformer().to(device)
    try:
        model.load_state_dict(torch.load('/kaggle/working/best_model.pt'))
        print("Best weights loaded successfully.", flush=True)
    except Exception as e:
        print(f"Loading failed ({e}). Benchmarking architecture only.", flush=True)
    
    model.eval()
    sample_input = torch.randn(1, 10, 40).to(device)
    
    print("Starting Warm-up...", flush=True)
    with torch.no_grad():
        for _ in range(50):
            _ = model(sample_input)
                    
        if device.type == 'cuda': torch.cuda.synchronize()
        print("Measuring 1000 iterations...", flush=True)
        
        start_time = time.perf_counter()
        for i in range(1000):
            _ = model(sample_input)
            if i % 250 == 0:
                print(f"  ... {i}/1000 complete", flush=True)
                
        if device.type == 'cuda': torch.cuda.synchronize()
        end_time = time.perf_counter()
    
    avg_ms = ((end_time - start_time) / 1000) * 1000
    print(f"\nRESULT: {avg_ms:.4f} ms per tick")
    return avg_ms

#EXECUTE
latency_result = run_final_benchmark()
