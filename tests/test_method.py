"""Equation and training-wiring tests; TestMixer does not validate Mamba kernels."""

import importlib.util
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.modules import EdgeDetector, FocusGatingUnit, SpectralEmbedding, FG_Mamba_Block
from models.fg_mamba import FG_Mamba
from utils.loss import FocalLoss, FocalOrthogonalLoss
from utils.dataset import HSIPatchDataset, sample_train_test_dsformer
from main import compute_class_weights
from train import train_epoch, evaluate


class TestMixer(nn.Module):
    """Deterministic causal stand-in used only to test scan/fusion wiring."""

    def __init__(self, d_model, **kwargs):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model))

    def forward(self, sequence):
        return torch.tanh(sequence.cumsum(dim=1) * self.scale)


def test_mixer():
    return patch.dict(sys.modules, {'mamba_ssm': types.SimpleNamespace(Mamba=TestMixer)})


class MethodTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_embedding_initialization(self):
        embedding = SpectralEmbedding(8, 4)
        x = torch.randn(2, 8, 5, 5)
        expected = embedding.norm(embedding.proj(x).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        torch.testing.assert_close(embedding(x, torch.randn(2, 1, 5, 5)), expected)
        self.assertIsNone(embedding.proj.bias)

    def test_gaussian_schedule_and_relative_boundary_gate(self):
        for layer, sigma in enumerate(np.linspace(25 / 4, 25 / 8, 6)):
            gate = FocusGatingUnit(25, layer, 6)
            self.assertAlmostEqual(gate.gpe_map[0, 0, 12, 13].item(), np.exp(-1 / (2 * sigma**2)), places=6)
            self.assertEqual(gate.gpe_map[0, 0, 12, 12].item(), 1.0)
        gate = FocusGatingUnit(5, 0, 2)
        gate.alpha.data.zero_()
        gate.edge_suppress.data.zero_()
        edges = torch.zeros(1, 1, 5, 5)
        edges[..., 2, 1] = 1.0
        result = gate(torch.ones(1, 3, 5, 5), edges)
        expected = (0.5 * gate.gpe_map + 0.5) * (1 - 0.5 * edges.sigmoid())
        torch.testing.assert_close(result, expected.expand_as(result))
        self.assertLess(result[0, 0, 2, 1], result[0, 0, 2, 3])

    def test_sobel_samplewise_normalization(self):
        detector = EdgeDetector()
        edges = detector(torch.ones(2, 1, 5, 5))
        torch.testing.assert_close(edges, torch.zeros_like(edges))
        x = torch.randn(1, 1, 5, 5)
        together = detector(torch.cat([x, 7 * x], dim=0))
        torch.testing.assert_close(together[:1], detector(x))
        self.assertGreaterEqual(together.min().item(), 0)
        self.assertLessEqual(together.max().item(), 1)

    def test_direction_scans_restore_and_pixelwise_fusion(self):
        with test_mixer():
            block = FG_Mamba_Block(4, 5, total_layers=2)
        x = torch.randn(2, 4, 5, 5)
        edge = torch.rand(2, 1, 5, 5)
        normalized = block.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        z = block.gating_unit(normalized, edge)
        row = [(i, j) for i in range(5) for j in range(5)]
        col = [(i, j) for j in range(5) for i in range(5)]
        restored = []
        for indices in (row, row[::-1], col, col[::-1]):
            sequence = torch.stack([z[:, :, i, j] for i, j in indices], dim=1)
            values = block.mixer(sequence)
            grid = torch.zeros_like(z)
            for index, (i, j) in enumerate(indices):
                grid[:, :, i, j] = values[:, index]
            restored.append(grid)
        torch.testing.assert_close(block(x, edge), x + sum(restored) / 4)
        nn.init.normal_(block.direction_gate.weight)
        weights = block.direction_gate(torch.cat(restored, dim=1)).softmax(dim=1)
        torch.testing.assert_close(weights.sum(1), torch.ones(2, 5, 5))
        expected = x + sum(weights[:, k:k + 1] * restored[k] for k in range(4))
        torch.testing.assert_close(block(x, edge), expected)

    def test_normalized_center_context_head(self):
        with test_mixer():
            model = FG_Mamba(8, 3, patch_size=5, embed_dim=4, depth=2)
        raw, pca = torch.randn(2, 8, 5, 5), torch.randn(2, 1, 5, 5)
        features = []
        handles = [layer.register_forward_hook(lambda m, i, o: features.append(o)) for layer in model.layers]
        logits = model(raw, pca)
        for handle in handles:
            handle.remove()
        centers = torch.stack([norm(f[:, :, 2, 2]) for norm, f in zip(model.center_token_norms, features)], 1)
        context = model.context_norm((features[-1] * model.gpe_cls).sum((2, 3)))
        torch.testing.assert_close(logits, model.cls_head(0.5 * centers.mean(1) + 0.5 * context))
        self.assertAlmostEqual(model.gpe_cls.sum().item(), 1.0, places=6)
        self.assertIsNot(model.layers[0].mixer, model.layers[1].mixer)

    def test_focal_and_orthogonality_equations(self):
        logits = torch.tensor([[0.2, -0.1, 1.0], [1.2, 0.0, -0.5]], requires_grad=True)
        labels = torch.tensor([2, 0])
        weights = compute_class_weights(np.array([0, 0, 1, 2, 2, 2]), 3)
        self.assertAlmostEqual(np.mean(weights), 1)
        p = logits.softmax(1)[torch.arange(2), labels]
        expected = -(torch.tensor(weights)[labels] * (1 - p)**2 * p.log()).mean()
        torch.testing.assert_close(FocalLoss(3, weights)(logits, labels), expected)
        projection = torch.eye(3, 5).reshape(3, 5, 1, 1).requires_grad_()
        criterion = FocalOrthogonalLoss(3, class_weights=weights)
        total, parts = criterion(logits, labels, projection)
        self.assertEqual(parts['ortho'], 0)
        torch.testing.assert_close(total, expected)
        doubled = projection * 2
        self.assertAlmostEqual(criterion.projection_orthogonality(doubled).item(), 9)
        extreme = torch.tensor([[1000., -1000.]], requires_grad=True)
        hard_loss = FocalLoss(2)(extreme, torch.tensor([1]))
        hard_loss.backward()
        self.assertTrue(torch.isfinite(hard_loss))
        self.assertGreater(extreme.grad.abs().sum().item(), 0)

    def test_training_evaluation_and_checkpoint_wiring(self):
        with test_mixer():
            model = FG_Mamba(8, 3, patch_size=5, embed_dim=4, depth=2)
        data = TensorDataset(torch.randn(6, 8, 5, 5), torch.randn(6, 1, 5, 5), torch.tensor([0, 1, 2, 0, 1, 2]))
        loader = DataLoader(data, batch_size=3)
        args = types.SimpleNamespace(epochs=1, grad_clip=0., amp=False)
        initial = model.embedding.proj.weight.detach().clone()
        losses = train_epoch(model, loader, FocalOrthogonalLoss(3), torch.optim.AdamW(model.parameters()), torch.device('cpu'), 1, args)
        self.assertTrue(all(np.isfinite(v) for v in losses.values()))
        self.assertFalse(torch.equal(initial, model.embedding.proj.weight))
        self.assertIsNotNone(model.layers[0].gating_unit.alpha.grad)
        result = evaluate(model, loader, torch.device('cpu'), args)
        self.assertEqual(result['Confusion matrix'].sum(), 6)
        with test_mixer():
            restored = FG_Mamba(8, 3, patch_size=5, embed_dim=4, depth=2)
        restored.load_state_dict(model.state_dict(), strict=True)
        torch.testing.assert_close(model(data.tensors[0], data.tensors[1]), restored(data.tensors[0], data.tensors[1]))

    def test_split_and_paired_augmentation(self):
        gt = np.repeat([0, 1, 2], [46, 80, 20]).reshape(1, -1)
        train, test = sample_train_test_dsformer(gt, 50, 202401)
        self.assertEqual([(train == k).sum() for k in range(3)], [15, 50, 15])
        self.assertFalse(np.any((train >= 0) & (test >= 0)))
        np.testing.assert_array_equal(np.where(train >= 0, train, test), gt)
        raw = np.arange(49, dtype=np.float32).reshape(7, 7, 1)
        labels = np.full((7, 7), -1); labels[3, 3] = 0
        dataset = HSIPatchDataset(raw, labels, 5, pca_image=raw.copy(), data_aug=True)
        for flip in range(3):
            for rotation in range(4):
                with patch('numpy.random.randint', side_effect=[flip, rotation]):
                    full, pca, label = dataset[0]
                torch.testing.assert_close(full, pca)
                self.assertEqual(full[0, 2, 2].item(), raw[3, 3, 0])

    @unittest.skipUnless(importlib.util.find_spec('mamba_ssm') and torch.cuda.is_available(), 'Requires real mamba-ssm and CUDA')
    def test_real_mamba_cuda_forward_backward(self):
        model = FG_Mamba(16, 3, patch_size=5, embed_dim=8, depth=2).cuda()
        logits = model(torch.randn(2, 16, 5, 5, device='cuda'), torch.randn(2, 1, 5, 5, device='cuda'))
        loss, _ = FocalOrthogonalLoss(3).cuda()(logits, torch.tensor([0, 1], device='cuda'), model.embedding.proj.weight)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.layers[0].mixer.in_proj.weight.grad)


if __name__ == '__main__':
    unittest.main()
