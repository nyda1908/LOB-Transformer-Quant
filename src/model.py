import torch
import torch.nn as nn
import math

class LiquidTransformer(nn.Module):
    def __init__(self, input_dim=40, d_model=64, nhead=4, num_layers=2, num_classes=3):
        super(LiquidTransformer, self).__init__()
        
        # ... (keep your existing embeddings and transformer layers) ...

        # 4. UPDATED NOVELTY PARAMETERS
        # A corresponds to self-leakage in the paper 
        self.A = nn.Parameter(torch.rand(d_model)) 
        self.dt = 0.1 # Integration step size
        self.input_weight = nn.Parameter(torch.rand(d_model))
        self.classifier = nn.Linear(d_model, num_classes)

    # --- PLACE THE DEF HERE ---
    def liquid_step(self, h, input_signal):
        """
        Implements the ODE dx/dt = -[A + S(t)]x(t) + S(t)I(t) 
        """
        # input_signal acts as S(t) from your whitepaper 
        derivative = -(self.A + input_signal) * h + (input_signal * self.input_weight)
        return h + self.dt * derivative 

    def forward(self, x):
        x = self.embedding(x) + self.pos_encoder
        x = self.transformer_encoder(x) # [batch, seq_len, d_model]
        
        # --- NEW LOOP LOGIC ---
        # Initialize h with the first time step [cite: 34]
        h = x[:, 0, :] 
        
        # Iterate through the time-window (Width W in your grid) [cite: 34]
        for t in range(1, x.size(1)):
            h = self.liquid_step(h, x[:, t, :])
            
        # h now represents the 'evolved' liquid state [cite: 6, 64]
        return self.classifier(h)
