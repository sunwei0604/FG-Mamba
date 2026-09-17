"""Spectral embedding, structural cue, and FG-Mamba blocks (Eqs. 1-12)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralEmbedding(nn.Module):
    """Bias-free spectral projection and static/PCA-conditioned corrections."""

    def __init__(self, in_channels, embed_dim=128):
        super().__init__()
        if not 0 < embed_dim <= in_channels:
            raise ValueError("Projection orthogonality requires 0 < embed_dim <= in_channels.")
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm(embed_dim)
        self.static_band_embed = nn.Parameter(torch.zeros(1, embed_dim, 1, 1))
        self.dynamic_band_embed = nn.Conv2d(1, embed_dim, kernel_size=1, bias=True)
        nn.init.zeros_(self.dynamic_band_embed.weight)
        nn.init.zeros_(self.dynamic_band_embed.bias)

    def forward(self, x, x_pca):
        feat = self.proj(x) + self.static_band_embed
        feat = feat + torch.tanh(self.dynamic_band_embed(x_pca))
        return self.norm(feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class EdgeDetector(nn.Module):
    """Fixed reflect-padded Sobel magnitude, min-max normalized per patch."""

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('filter_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('filter_y', sobel_y.view(1, 1, 3, 3))

    def forward(self, x_pca):
        padded = F.pad(x_pca, (1, 1, 1, 1), mode='reflect')
        g_x = F.conv2d(padded, self.filter_x)
        g_y = F.conv2d(padded, self.filter_y)
        edge_map = torch.sqrt(g_x.square() + g_y.square() + 1e-6)
        minimum = edge_map.amin(dim=(2, 3), keepdim=True)
        maximum = edge_map.amax(dim=(2, 3), keepdim=True)
        return (edge_map - minimum) / (maximum - minimum + 1e-6)


def gaussian_map(patch_size, sigma):
    """Fixed Gaussian spatial prior; zero-based center equals (P-1)/2."""
    if patch_size < 3 or patch_size % 2 == 0:
        raise ValueError("patch_size must be an odd integer >= 3.")
    coords = torch.arange(patch_size, dtype=torch.float32) - patch_size // 2
    y, x = torch.meshgrid(coords, coords, indexing='ij')
    return torch.exp(-(x.square() + y.square()) / (2 * sigma ** 2)).view(
        1, 1, patch_size, patch_size
    )


class FocusGatingUnit(nn.Module):
    """Depth-dependent Gaussian and center-relative boundary gates (Eqs. 4-9)."""

    def __init__(self, patch_size=25, layer_idx=0, total_layers=6):
        super().__init__()
        if total_layers < 1 or not 0 <= layer_idx < total_layers:
            raise ValueError("Require total_layers >= 1 and 0 <= layer_idx < total_layers.")
        sigma = (patch_size / 4.0) * (1.0 - 0.5 * layer_idx / max(total_layers - 1, 1))
        self.register_buffer('gpe_map', gaussian_map(patch_size, sigma))
        # Retain the existing weak-gating initialization; unspecified in the paper.
        self.alpha = nn.Parameter(torch.tensor(-4.0))
        self.edge_suppress = nn.Parameter(torch.tensor(-4.0))

    def forward(self, x, edge_map):
        alpha = torch.sigmoid(self.alpha)
        center_gate = alpha * self.gpe_map + (1.0 - alpha)
        h, w = x.shape[-2:]
        center_edge = edge_map[:, :, h // 2:h // 2 + 1, w // 2:w // 2 + 1]
        relative_response = torch.sigmoid(edge_map - center_edge)
        boundary_gate = 1.0 - torch.sigmoid(self.edge_suppress) * relative_response
        return x * (center_gate * boundary_gate)


class FG_Mamba_Block(nn.Module):
    """Pre-normalized gating, shared four-path Mamba, and residual fusion."""

    def __init__(self, dim, patch_size=25, layer_idx=0, total_layers=6):
        super().__init__()
        from mamba_ssm import Mamba

        self.norm = nn.LayerNorm(dim)
        self.gating_unit = FocusGatingUnit(patch_size, layer_idx, total_layers)
        self.mixer = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)
        self.direction_gate = nn.Conv2d(dim * 4, 4, kernel_size=1, bias=True)
        nn.init.zeros_(self.direction_gate.weight)
        nn.init.zeros_(self.direction_gate.bias)

    def forward(self, x, edge_map):
        b, c, h, w = x.shape
        normalized = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        gated = self.gating_unit(normalized, edge_map)

        row = gated.flatten(2).permute(0, 2, 1)
        column = gated.transpose(2, 3).flatten(2).permute(0, 2, 1)
        # Directions share one mixer within each block, not across blocks.
        sequences = torch.cat([row, row.flip([1]), column, column.flip([1])], dim=0)
        out1, out2, out3, out4 = self.mixer(sequences).chunk(4, dim=0)

        # Invert each scan before computing pixel-wise direction weights.
        out1 = out1.permute(0, 2, 1).reshape(b, c, h, w)
        out2 = out2.flip([1]).permute(0, 2, 1).reshape(b, c, h, w)
        out3 = out3.permute(0, 2, 1).reshape(b, c, w, h).transpose(2, 3)
        out4 = out4.flip([1]).permute(0, 2, 1).reshape(b, c, w, h).transpose(2, 3)
        outputs = (out1, out2, out3, out4)
        weights = torch.softmax(self.direction_gate(torch.cat(outputs, dim=1)), dim=1)
        fused = sum(weights[:, k:k + 1] * output for k, output in enumerate(outputs))
        return x + fused
