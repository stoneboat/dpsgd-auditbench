"""ScatterLinear: kymatio scattering features + linear classifier.

Drop-in replacement for `WideResNet` when running DP-FTRL audits with the
Tramer-Boneh handcrafted-feature recipe. The scattering transform has no
learnable parameters, so under Opacus' GradSampleModule only the trailing
Linear layer produces per-sample gradients -- exactly what we need for a
clean DP-FTRL audit.

Install:  pip install kymatio
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from kymatio.torch import Scattering2D
except ImportError as e:
    raise ImportError(
        "ScatterLinear requires the `kymatio` package. Install with: "
        "pip install kymatio"
    ) from e


class ScatterLinear(nn.Module):
    """Scattering features (J scales) + GroupNorm + Linear classifier.

    For 32x32 RGB CIFAR-10 with J=2, the scattering output is
    (B, 3, 81, 8, 8) with the kymatio default L=8 angles. We collapse
    the (3, 81) axes to a single 243-channel axis so GroupNorm can
    standardize per (channel-group, sample) -- no batch coupling, fully
    Opacus-compatible. feat_dim = 3 * 81 * 64 = 15552.
    """

    def __init__(self, num_classes: int = 10, image_size: int = 32, J: int = 2,
                 num_groups: int = 27):
        super().__init__()
        self.J = J
        self.image_size = image_size
        self.num_classes = num_classes

        self.scattering = Scattering2D(J=J, shape=(image_size, image_size))

        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size)
            feat = self.scattering(dummy)
        # Expected scattering shape: (B, C_in, K, H', W'); flatten (C_in, K) -> channels.
        if feat.ndim != 5:
            raise ValueError(f"Unexpected scattering output ndim={feat.ndim}, shape={tuple(feat.shape)}")
        self._scatter_channels = int(feat.shape[1] * feat.shape[2])
        self._scatter_hw = (int(feat.shape[3]), int(feat.shape[4]))
        self.feat_dim = int(feat.numel())

        if self._scatter_channels % num_groups != 0:
            raise ValueError(
                f"num_groups={num_groups} must divide channels={self._scatter_channels}"
            )
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=self._scatter_channels)
        self.linear = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.scattering(x)                                    # (B, C_in, K, H, W)
        B = feat.shape[0]
        feat = feat.reshape(B, self._scatter_channels, *self._scatter_hw)
        feat = self.gn(feat)
        feat = feat.reshape(B, -1)
        return self.linear(feat)