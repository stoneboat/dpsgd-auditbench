#!/usr/bin/env python3
"""Real DP-SGD run with an input-space canary inserted, instrumented for Appendix E.

Training is the audited path itself (train.train_whitebox); this script only supplies
the canary, the dataset it lives in, and a probe. At every logical step the probe logs
the two quantities the reduction to Model 1 needs:

  Assumption 2 (exact saturation):  a_t = ||clip_C(grad canary)|| == C, i.e. ||g_t|| >= C.
  Assumption 1 (background suppression):  h_{t,z} = <u_t, clip_C(grad z)> == 0,
      reported as batch sums sum_h / sum_h2 and their 1/q-scaled estimates of the
      sums over D that Proposition 3 writes.

u_t is the canary's own unit direction at theta_{t-1}, recomputed every step --
Proposition 4 allows it to move arbitrarily, so stability is not required.

Usage:
  python scripts/canary_saturation_run.py --epsilon 8 --target-steps 500 \
      --canary-color 1,0,0 --canary-label 6 --out canary_sat.csv
  python scripts/canary_saturation_run.py --epsilon 8 --target-steps 500 --dirac 0 ...
"""
import argparse
import json
import math
import os
import secrets
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from opacus import PrivacyEngine

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(os.path.join(project_dir, 'src'))

from network_arch import WideResNet  # noqa: E402
from train import _augment_per_image, train_whitebox  # noqa: E402
from utils import setup_logging  # noqa: E402

NORMALIZE = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))


class ConstantCanary(torch.utils.data.Dataset):
    """Single constant-color image with a fixed label; exactly invariant to the
    RandomCrop(reflect)+flip augmentation, so all K augs share one gradient."""

    def __init__(self, rgb, label):
        self.img = torch.tensor(rgb, dtype=torch.float32).view(3, 1, 1).expand(3, 32, 32).contiguous()
        self.label = int(label)

    def __len__(self):
        return 1

    def __getitem__(self, i):
        return self.img.clone(), self.label


def dirac_direction(params, flat_idx):
    """Unit vector e_c over the trainable params, c given as a global flat index."""
    u, off = [], 0
    for p in params:
        t = torch.zeros_like(p)
        if off <= flat_idx < off + p.numel():
            t.view(-1)[flat_idx - off] = 1.0
        off += p.numel()
        u.append(t)
    if not (0 <= flat_idx < off):
        raise ValueError(f"--dirac {flat_idx} out of range (total coords={off})")
    return u


class SaturationProbe:
    """begin_step / observe / end_step hooks called by train_whitebox."""

    def __init__(self, optimizer, canary_img, label, C, q, K, device, dirac=None, logger=None, log_every=10):
        self.opt = optimizer
        self.img = canary_img
        self.label = label
        self.C = C
        self.q = q
        self.K = K
        self.device = device
        self.dirac = dirac
        self.logger = logger
        self.log_every = log_every
        self.rows = []
        self.u = None
        self.g_norm = float('nan')
        self._reset_accumulators()

    def _reset_accumulators(self):
        self.acc_h = self.acc_h2 = 0.0
        self.acc_n = 0

    def begin_step(self, model):
        if self.dirac is not None:
            self.u, self.g_norm = dirac_direction(self.opt.params, self.dirac), float('inf')
        else:
            self.g_norm, self.u = self._canary_gradient(model)
        self._reset_accumulators()

    def _canary_gradient(self, model):
        """Per-example canary gradient at the current iterate, aug-averaged like training.
        Hooks off, so grad_sample is untouched and p.grad is exactly what the clipper sees."""
        model.disable_hooks()
        model.zero_grad(set_to_none=True)
        x = self.img.unsqueeze(0).to(self.device)
        if self.K > 1:
            x = torch.stack([_augment_per_image(x)[0] for _ in range(self.K)])
        x = NORMALIZE(x)
        y = torch.full((x.shape[0],), self.label, dtype=torch.long, device=self.device)
        nn.functional.cross_entropy(model(x), y, reduction='mean').backward()
        g = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in self.opt.params]
        model.zero_grad(set_to_none=True)
        model.enable_hooks()
        norm = math.sqrt(sum(float((t * t).sum()) for t in g))
        return norm, ([t / norm for t in g] if norm > 0 else g)

    def observe(self, optimizer):
        """sum_z h and sum_z h^2 over background records in this physical batch."""
        dots = norms_sq = None
        for p, up in zip(optimizer.params, self.u):
            gs = p.grad_sample
            if gs is None:
                continue
            flat = gs.reshape(gs.shape[0], -1)
            d = flat @ up.reshape(-1)
            s = (flat * flat).sum(dim=1)
            dots = d if dots is None else dots + d
            norms_sq = s if norms_sq is None else norms_sq + s
        if dots is None:
            return
        norms = norms_sq.clamp_min(1e-12).sqrt()
        h = (self.C / norms).clamp_max(1.0) * dots          # <u, clip_C(g_z)>
        if self.dirac is None:
            # Drop the canary's own row: it is the only record parallel to u_t.
            h = h[(dots / norms).abs() < 0.999]
        self.acc_h += float(h.sum())
        self.acc_h2 += float((h * h).sum())
        self.acc_n += int(h.numel())

    def end_step(self, optimizer):
        t = len(self.rows) + 1
        self.rows.append(dict(step=t, g_norm=self.g_norm, ratio=self.g_norm / self.C,
                              saturated=int(self.g_norm >= self.C), sum_h=self.acc_h, sum_h2=self.acc_h2,
                              n_bg=self.acc_n, mean_h=self.acc_h / max(1, self.acc_n),
                              sum_h_D=self.acc_h / self.q, sum_h2_D=self.acc_h2 / self.q))
        if self.logger is not None and (t % self.log_every == 0 or t == 1):
            self.logger.info(f"step {t:5d} ||g||/C={self.g_norm / self.C:8.3f} sum_h={self.acc_h:+.4f} "
                             f"mean_h={self.acc_h / max(1, self.acc_n):+.6f} sum_h2={self.acc_h2:.4f} n_bg={self.acc_n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--canary-color', default='1,0,0', help='RGB in [0,1], comma separated')
    ap.add_argument('--canary-label', type=int, default=6)
    ap.add_argument('--dirac', type=int, default=None, metavar='FLAT_IDX',
                    help='baseline arm: score along dirac e_c and insert no input canary. '
                         'A2 is exact by construction; this measures A1 for the dirac audit.')
    ap.add_argument('--epsilon', type=float, default=8.0)
    ap.add_argument('--delta', type=float, default=1e-5)
    ap.add_argument('--target-steps', type=int, default=500)
    ap.add_argument('--logical-batch-size', type=int, default=4096)
    ap.add_argument('--max-physical-batch-size', type=int, default=128)
    ap.add_argument('--aug-multiplicity', type=int, default=16)
    ap.add_argument('--max-grad-norm', type=float, default=1.0)
    ap.add_argument('--lr', type=float, default=4.0)
    ap.add_argument('--momentum', type=float, default=0.0)
    ap.add_argument('--seed', type=str, default=None)
    ap.add_argument('--data-dir', default='./data')
    ap.add_argument('--log-dir', default='./logs')
    ap.add_argument('--log-file', default=None)
    ap.add_argument('--out', default='canary_saturation.csv')
    ap.add_argument('--device', default='auto')
    ap.add_argument('--num-workers', type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    logger, _ = setup_logging(log_file=args.log_file, log_dir=args.log_dir)

    seed = int(args.seed) if args.seed else secrets.randbits(64)
    torch.manual_seed(seed % (2**32 - 1))
    np.random.seed(seed % (2**32 - 1))
    device = torch.device(args.device if args.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))
    C = args.max_grad_norm
    rgb = [float(v) for v in args.canary_color.split(',')]
    arm = f"dirac coord {args.dirac}" if args.dirac is not None else f"const{rgb} label {args.canary_label}"
    logger.info(f"device={device} seed={seed} arm={arm} C={C} K={args.aug_multiplicity}")

    base = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transforms.ToTensor())
    canary_ds = ConstantCanary(rgb, args.canary_label)
    train_ds = base if args.dirac is not None else torch.utils.data.ConcatDataset([base, canary_ds])
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.logical_batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    logger.info(f"dataset: {len(base)} CIFAR + {0 if args.dirac is not None else 1} canary = {len(train_ds)}")

    model = WideResNet(depth=16, widen_factor=4).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    steps_per_epoch = max(1, len(train_loader))
    epochs_for_priv = max(1, math.ceil(args.target_steps / steps_per_epoch))
    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(module=model, optimizer=optimizer, data_loader=train_loader, epochs=epochs_for_priv, target_epsilon=args.epsilon, target_delta=args.delta, max_grad_norm=C)
    sigma = optimizer.noise_multiplier
    q = 1.0 / steps_per_epoch   # Poisson sample rate = logical_batch / N
    tau = sigma * C
    logger.info(f"sigma={sigma:.4f} q={q:.6f} tau=sigma*C={tau:.4f} epochs_for_priv={epochs_for_priv} steps/epoch={steps_per_epoch}")

    probe = SaturationProbe(optimizer, canary_ds.img, args.canary_label, C, q, args.aug_multiplicity, device, dirac=args.dirac, logger=logger)

    steps_done = 0
    for epoch in range(1, epochs_for_priv + 1):
        if steps_done >= args.target_steps:
            break
        loss, n_steps = train_whitebox(model, optimizer, train_loader, device, epoch,
                                       aug_multiplicity=args.aug_multiplicity,
                                       max_physical_batch_size=args.max_physical_batch_size,
                                       logical_batch_size=args.logical_batch_size,
                                       canary_prob=0.0, probe=probe,
                                       max_logical_steps=args.target_steps - steps_done)
        steps_done = len(probe.rows)
        logger.info(f"epoch {epoch}: loss={loss:.4f} steps={steps_done}/{args.target_steps}")

    rows = probe.rows
    ratios = np.array([r['ratio'] for r in rows])
    mean_h = np.array([r['mean_h'] for r in rows])
    sum_h = np.array([r['sum_h_D'] for r in rows])
    sum_h2 = np.array([r['sum_h2_D'] for r in rows])
    drift, var_term = q * abs(sum_h.mean()), q * (1 - q) * sum_h2.mean()
    logger.info("=" * 70)
    if args.dirac is not None:
        logger.info(f"A2 saturation: exact by construction (dirac coord {args.dirac}), {len(rows)} steps")
    else:
        logger.info(f"A2 saturation: min ||g||/C = {ratios.min():.3f} over {len(rows)} steps; "
                    f"frac >= C = {np.mean(ratios >= 1.0):.4f}; argmin at step {rows[int(ratios.argmin())]['step']}")
        logger.info(f"A2 {'HOLDS' if ratios.min() >= 1.0 else 'FAILS'} on this run.")
    logger.info(f"A1 per-record alignment: mean_z h = {mean_h.mean():+.6f} (|clip_C(g_z)| <= C = {C})")
    logger.info(f"A1 drift:    q*sum_h    = {drift:+.6f}  vs canary q*C      = {q * C:.6f}")
    logger.info(f"A1 variance: q(1-q)*sum_h2 = {var_term:.6f}  vs canary q(1-q)C^2 = {q * (1 - q) * C * C:.6f}  vs tau^2 = {tau ** 2:.6f}")
    logger.info(f"A1 {'HOLDS (both terms < 5% of their canary/noise references)' if drift < 0.05 * q * C and var_term < 0.05 * tau ** 2 else 'FAILS: background projections are not negligible'}")

    with open(args.out, 'w') as f:
        f.write(','.join(rows[0].keys()) + '\n')
        for r in rows:
            f.write(','.join(str(v) for v in r.values()) + '\n')
    meta = dict(seed=seed, sigma=sigma, q=q, tau=tau, C=C, arm=arm, canary_color=rgb, canary_label=args.canary_label,
                dirac=args.dirac, epsilon=args.epsilon, delta=args.delta, steps=len(rows),
                aug_multiplicity=args.aug_multiplicity, lr=args.lr, target_steps=args.target_steps,
                min_ratio=float(ratios.min()), frac_saturated=float(np.mean(ratios >= 1.0)),
                drift=float(drift), var_term=float(var_term))
    with open(os.path.splitext(args.out)[0] + '_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    logger.info(f"wrote {args.out}")


if __name__ == '__main__':
    main()
