"""Full FG-Mamba classifier, including cross-layer center-context fusion."""

import torch
import torch.nn as nn
from .modules import SpectralEmbedding, EdgeDetector, FG_Mamba_Block, gaussian_map


class FG_Mamba(nn.Module):
    def __init__(self, hsi_channels, num_classes, patch_size=25,
                 embed_dim=128, depth=6):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1.")
        self.patch_size = patch_size
        self.depth = depth
        self.embedding = SpectralEmbedding(hsi_channels, embed_dim)
        self.edge_detector = EdgeDetector()
        self.layers = nn.ModuleList([
            FG_Mamba_Block(embed_dim, patch_size, layer_idx=i, total_layers=depth)
            for i in range(depth)
        ])
        self.cls_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes))
        self.center_token_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(depth)])
        self.center_fusion_weights = nn.Parameter(torch.zeros(depth))
        self.center_fusion_scorer = nn.Sequential(
            nn.Linear(embed_dim, max(embed_dim // 4, 16)),
            nn.GELU(),
            nn.Linear(max(embed_dim // 4, 16), 1),
        )
        nn.init.zeros_(self.center_fusion_scorer[-1].weight)
        nn.init.zeros_(self.center_fusion_scorer[-1].bias)
        self.context_norm = nn.LayerNorm(embed_dim)
        self.center_gate = nn.Parameter(torch.zeros(1))
        pool = gaussian_map(patch_size, patch_size / 8.0)
        self.register_buffer('gpe_cls', pool / pool.sum())

    def forward(self, x_raw, x_pca):
        if x_raw.ndim != 4 or x_raw.shape[-2:] != (self.patch_size, self.patch_size):
            raise ValueError("x_raw must have shape (B, C, patch_size, patch_size).")
        if x_pca is None or x_pca.shape != (x_raw.shape[0], 1, self.patch_size, self.patch_size):
            raise ValueError("x_pca must have shape (B, 1, patch_size, patch_size).")
        x = self.embedding(x_raw, x_pca)
        edge_map = self.edge_detector(x_pca)
        center = self.patch_size // 2
        normalized_centers = []
        for layer, norm in zip(self.layers, self.center_token_norms):
            x = layer(x, edge_map)
            normalized_centers.append(norm(x[:, :, center, center]))

        # Both scoring and aggregation use normalized tokens (corrected Eq. 15).
        centers = torch.stack(normalized_centers, dim=1)
        scores = self.center_fusion_scorer(centers).squeeze(-1)
        weights = torch.softmax(scores + self.center_fusion_weights, dim=1)
        f_center = (centers * weights.unsqueeze(-1)).sum(dim=1)
        f_context = self.context_norm((x * self.gpe_cls).sum(dim=(2, 3)))
        alpha = torch.sigmoid(self.center_gate)
        return self.cls_head(alpha * f_center + (1.0 - alpha) * f_context)
