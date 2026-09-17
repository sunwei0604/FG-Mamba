import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):

    def __init__(self, class_num, alpha=None, gamma=2.0,
                 ignore_index=-1, size_average=True):
        super().__init__()
        self.class_num = class_num
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average
        if alpha is None:
            self.register_buffer('alpha', torch.ones(class_num))
        elif isinstance(alpha, (list, torch.Tensor)):
            alpha_t = torch.tensor(alpha, dtype=torch.float32) \
                if not isinstance(alpha, torch.Tensor) else alpha.float()
            self.register_buffer('alpha', alpha_t / alpha_t.mean())
        else:
            raise TypeError("alpha must be None, list, or torch.Tensor")

    def forward(self, inputs, targets):
        valid_mask = (targets != self.ignore_index)
        valid_inputs = inputs[valid_mask]
        valid_targets = targets[valid_mask]
        if valid_targets.numel() == 0:
            return valid_inputs.sum()
        # Eq. (19), using log-softmax to preserve gradients for hard examples.
        log_p = F.log_softmax(valid_inputs.float(), dim=1).gather(
            1, valid_targets.unsqueeze(1)
        ).squeeze(1)
        probs = log_p.exp()
        alpha_t = self.alpha[valid_targets]
        batch_loss = -alpha_t * ((1 - probs) ** self.gamma) * log_p
        return batch_loss.mean() if self.size_average else batch_loss.sum()


class FocalOrthogonalLoss(nn.Module):

    def __init__(self, num_classes, gamma=2.0, lambda_ortho=0.001,
                 ignore_index=-1, class_weights=None):
        super().__init__()
        self.lambda_ortho = lambda_ortho
        self.criterion = FocalLoss(
            num_classes,
            alpha=class_weights,
            gamma=gamma,
            ignore_index=ignore_index,
        )

    @staticmethod

    def projection_orthogonality(projection_weight):
        weight = projection_weight.flatten(1)
        d, channels = weight.shape
        if d > channels:
            raise ValueError(
                "Projection orthogonality requires d <= C, "
                f"got d={d}, C={channels}."
            )
        with torch.cuda.amp.autocast(enabled=False):
            weight_fp32 = weight.float()
            gram = weight_fp32 @ weight_fp32.transpose(0, 1)
            identity = torch.eye(d, dtype=weight_fp32.dtype, device=weight.device)
            return (gram - identity).square().sum() / d

    def forward(self, logits, labels, projection_weight):
        loss_main = self.criterion(logits, labels)
        loss_ortho = self.projection_orthogonality(projection_weight)
        total_loss = loss_main + self.lambda_ortho * loss_ortho
        return total_loss, {
            'main': loss_main.detach().item(),
            'ortho': loss_ortho.detach().item(),
        }
