import torch
import torch.nn as nn


class LTCCell(nn.Module):
    """Liquid Time-Constant cell: continuous-time hidden state via Euler-discretized ODE."""
    def __init__(self, input_size, hidden_size):
        super(LTCCell, self).__init__()
        self.hidden_size = hidden_size
        self.W = nn.Linear(input_size + hidden_size, hidden_size)
        self.A = nn.Parameter(torch.ones(hidden_size))      # leakage coefficient
        self.tau = nn.Parameter(torch.ones(hidden_size))    # time constant

    def forward(self, x, h):
        combined = torch.cat([x, h], dim=-1)
        S = torch.sigmoid(self.W(combined))
        # dh/dt = -[A + S(t)]h + S(t)x   (Euler step)
        dh = (-(self.A + S) * h + S * x) / self.tau
        return h + dh


class LiquidTransformer(nn.Module):
    """Multi-Channel Transformer with an LTC integration head."""
    def __init__(self, input_dim=40, d_model=64, nhead=4, num_layers=2, num_classes=3):
        super(LiquidTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True),
            num_layers=num_layers
        )
        self.ltc_cell = LTCCell(input_size=d_model, hidden_size=d_model)
        self.fc = nn.Linear(d_model, num_classes)
        self.d_model = d_model

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        h = torch.zeros(x.size(0), self.d_model).to(x.device)
        for t in range(x.size(1)):
            h = self.ltc_cell(x[:, t, :], h)
        return self.fc(h)


class PlainTransformer(nn.Module):
    """Ablation baseline: identical Transformer with a standard last-token readout (no LTC)."""
    def __init__(self, input_dim=40, d_model=64, nhead=4, num_layers=2, num_classes=3):
        super(PlainTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True),
            num_layers=num_layers
        )
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        return self.fc(x[:, -1, :])
