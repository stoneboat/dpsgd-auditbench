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
    """Scattering features (J scales) + Linear classifier.

    For 32x32 RGB CIFAR-10 with J=2, the scattering output is
    (B, 3, 81, 8, 8) with the kymatio default L=8 angles, giving
    feat_dim = 3 * 81 * 64 = 15552. The trailing Linear maps that to
    `num_classes` logits.
    """

    def __init__(self, num_classes: int = 10, image_size: int = 32, J: int = 2):
        super().__init__()
        self.J = J
        self.image_size = image_size
        self.num_classes = num_classes

        self.scattering = Scattering2D(J=J, shape=(image_size, image_size))

        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size)
            feat = self.scattering(dummy)
        self.feat_dim = int(feat.numel())

        self.linear = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.scattering(x)
        feat = feat.reshape(feat.size(0), -1)
        return self.linear(feat)