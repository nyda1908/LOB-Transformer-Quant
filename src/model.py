import torch
import torch.nn as nn
import math

class LiquidTransformer(nn.Module):
    def __init__(self, input_dim=40, d_model=64, nhead=4, num_layers=2, num_classes=3):
        super(LiquidTransformer, self).__init__()
        
        # 1. Multi-Channel Embedding
        # We project the 40 LOB features into a higher d_model space
        self.embedding = nn.Linear(input_dim, d_model)
        
        # 2. Positional Encoding
        # Adds time-awareness to the Transformer
        self.pos_encoder = nn.Parameter(torch.zeros(1, 100, d_model)) 
        
        # 3. Transformer Encoder (The Base Model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. THE NOVEL EXTENSION: Liquid Layer
        # Replacing the standard linear head with a continuous-time neuron
        self.liquid_head = nn.Linear(d_model, d_model)
        self.tau = nn.Parameter(torch.ones(d_model)) # Learnable time constant
        
        # 5. Prediction Head (Up, Down, Neutral)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len, 40)
        x = self.embedding(x) + self.pos_encoder
        
        # Pass through Transformer to capture Spatial-Temporal patterns
        x = self.transformer_encoder(x)
        
        # Apply the Liquid logic to the last time step
        # h_new = h_old + (1/tau) * (-h_old + input)
        # Here we simulate one step of the liquid update
        last_step = x[:, -1, :]
        liquid_out = last_step + (1.0 / (1.0 + self.tau)) * (-last_step + self.liquid_head(last_step))
        
        return self.classifier(liquid_out)
