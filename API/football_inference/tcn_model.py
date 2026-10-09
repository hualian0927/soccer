from __future__ import annotations

import torch
from torch import nn


class TCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.net(inputs)


class BidirectionalTrajectoryTCN(nn.Module):
    def __init__(self, feature_dim=16, hidden_dim=64, dilations=(1, 2, 4, 8, 16), dropout=0.1):
        super().__init__()
        self.project = nn.Conv1d(feature_dim, hidden_dim, 1)
        self.blocks = nn.Sequential(*(TCNBlock(hidden_dim, dilation, dropout) for dilation in dilations))
        self.trajectory_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 2))
        quality_input = hidden_dim + feature_dim + 4
        self.quality_head = nn.Sequential(nn.Linear(quality_input, 64), nn.GELU(), nn.Linear(64, 1))
        self.uncertainty_head = nn.Sequential(nn.Linear(quality_input, 64), nn.GELU(), nn.Linear(64, 2))

    @staticmethod
    def localize_coordinates(trajectory_features: torch.Tensor):
        positions = trajectory_features[:, :, 0:2]
        observed = trajectory_features[:, :, 7] > 0.5
        valid_position = trajectory_features[:, :, 15] > 0.5
        weights = observed.to(positions.dtype).unsqueeze(-1)
        counts = weights.sum(dim=1).clamp_min(1.0)
        origin = (positions * weights).sum(dim=1) / counts
        localized = trajectory_features.clone()
        localized[:, :, 0:2] = torch.where(
            valid_position.unsqueeze(-1), positions - origin.unsqueeze(1), positions
        )
        return localized, origin

    def forward(self, features: torch.Tensor, trajectory_features: torch.Tensor):
        center_index = features.shape[1] // 2
        localized, coordinate_origin = self.localize_coordinates(trajectory_features)
        encoded = self.blocks(self.project(localized.transpose(1, 2))).transpose(1, 2)
        state = encoded[:, center_index]
        relative_center = self.trajectory_head(state)
        trajectory_center = coordinate_origin + relative_center
        current = features[:, center_index]
        detected_center = current[:, 0:2]
        difference = detected_center - trajectory_center
        distance = torch.linalg.vector_norm(difference, dim=1, keepdim=True)
        current_local = current.clone()
        current_local[:, 0:2] = detected_center - coordinate_origin
        quality_features = torch.cat([state, current_local, difference, distance, current[:, 15:16]], dim=1)
        quality = torch.sigmoid(self.quality_head(quality_features)) * current[:, 7:8]
        self.uncertainty_head(quality_features)
        final_center = quality * detected_center + (1.0 - quality) * trajectory_center
        return {"trajectory_center": trajectory_center, "quality": quality, "final_center": final_center}
