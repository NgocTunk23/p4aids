"""Compact model registry for point and sequence anomaly detectors."""

from __future__ import annotations

import torch
from torch import nn

from .config import CONFIG, DLConfig

class MLP(nn.Module):
    def __init__(self, input_size: int, config: DLConfig = CONFIG):
        super().__init__()
        h, drop = config.model.hidden_size, config.model.dropout
        self.net = nn.Sequential(
            nn.Linear(input_size, h * 2), nn.LayerNorm(h * 2), nn.GELU(), nn.Dropout(drop),
            nn.Linear(h * 2, h), nn.GELU(), nn.Dropout(drop), nn.Linear(h, 1),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

class Chomp1d(nn.Module):
    def __init__(self, amount: int):
        super().__init__(); self.amount = amount
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., :-self.amount] if self.amount else x

class ResidualTCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel - 1) * dilation
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel, padding=padding, dilation=dilation),
            Chomp1d(padding), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel, padding=padding, dilation=dilation),
            Chomp1d(padding), nn.GELU(), nn.Dropout(dropout),
        )
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.norm = nn.GroupNorm(1, out_channels)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.block(x) + self.skip(x))

class TCN(nn.Module):
    def __init__(self, input_size: int, target_length: int, config: DLConfig = CONFIG):
        super().__init__()
        channels, kernel = config.model.tcn_channels, config.model.tcn_kernel_size
        layers, incoming = [], input_size
        for dilation in config.model.tcn_dilations:
            layers.append(ResidualTCNBlock(incoming, channels, kernel, dilation, config.model.dropout))
            incoming = channels
        self.network, self.head, self.target_length = nn.Sequential(*layers), nn.Conv1d(channels, 1, 1), target_length
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.head(self.network(x.transpose(1, 2))).squeeze(1)
        return logits[:, -self.target_length:]

class RecurrentLabeler(nn.Module):
    def __init__(self, input_size: int, target_length: int, cell: str, config: DLConfig = CONFIG):
        super().__init__()
        cls = nn.GRU if cell == "gru" else nn.LSTM
        self.rnn = cls(input_size, config.model.hidden_size, num_layers=config.model.num_layers,
                       dropout=config.model.dropout if config.model.num_layers > 1 else 0.0,
                       batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(config.model.dropout), nn.Linear(config.model.hidden_size * 2, 1))
        self.target_length = target_length
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.rnn(x)
        return self.head(encoded[:, -self.target_length:]).squeeze(-1)

class LSTMAutoencoder(nn.Module):
    """Sequence autoencoder; timestep reconstruction MSE is the anomaly score."""
    def __init__(self, input_size: int, target_length: int, config: DLConfig = CONFIG):
        super().__init__()
        h = config.model.hidden_size
        self.encoder = nn.LSTM(input_size, h, batch_first=True)
        self.decoder = nn.LSTM(h, h, batch_first=True)
        self.output = nn.Linear(h, input_size)
        self.target_length = target_length
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encoder(x)
        decoded, _ = self.decoder(encoded)
        return self.output(decoded)[:, -self.target_length:]

def build_model(name: str, input_size: int, config: DLConfig = CONFIG) -> nn.Module:
    target = config.data.target_length
    if name == "mlp": return MLP(input_size, config)
    if name == "tcn": return TCN(input_size, target, config)
    if name == "bigru": return RecurrentLabeler(input_size, target, "gru", config)
    if name == "bilstm": return RecurrentLabeler(input_size, target, "lstm", config)
    if name == "lstm_ae": return LSTMAutoencoder(input_size, target, config)
    raise ValueError(f"Unknown model {name!r}")

MODEL_REGISTRY = {name: build_model for name in ("mlp", "tcn", "bigru", "bilstm", "lstm_ae")}
