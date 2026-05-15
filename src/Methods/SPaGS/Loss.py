# -- coding: utf-8 --

"""SPaGS/Loss.py: Loss function."""

import torch
import torchmetrics

from Framework import ConfigParameterList
from Optim.Losses.Base import BaseLoss
from Optim.Losses.FusedDSSIM import fused_dssim


def ws_l1_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Equirectangular latitude-weighted L1 loss.

    Weights each pixel row by sin(π(i+0.5)/H) to match solid angle,
    so polar rows (which oversample 3D space) contribute less to the loss.
    Weights are normalized to mean=1 so the overall loss scale is unchanged.
    """
    H = input.shape[-2]
    i = torch.arange(H, device=input.device, dtype=torch.float32)
    ws = torch.sin(torch.pi * (i + 0.5) / H)   # [H]  solid-angle weight per row
    ws = (ws / ws.mean())[:, None]              # [H, 1]  mean-normalized
    return ((input - target).abs() * ws[None]).mean()


def standard_l1_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (input - target).abs().mean()


class SPaGSLoss(BaseLoss):
    def __init__(self, loss_config: ConfigParameterList, use_ws_loss: bool = True) -> None:
        super().__init__()
        l1_fn = ws_l1_loss if use_ws_loss else standard_l1_loss
        self.addLossMetric('L1_Color', l1_fn, loss_config.LAMBDA_L1)
        self.addLossMetric('DSSIM_Color', fused_dssim, loss_config.LAMBDA_DSSIM)
        self.addQualityMetric('PSNR', torchmetrics.functional.image.peak_signal_noise_ratio)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return super().forward({
            'L1_Color': {'input': input, 'target': target},
            'DSSIM_Color': {'input': input, 'target': target},
            'PSNR': {'preds': input, 'target': target, 'data_range': 1.0}
        })
