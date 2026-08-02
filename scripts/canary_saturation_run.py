#!/usr/bin/env python3
"""Real DP-SGD run with an input-space canary inserted, instrumented for Appendix E.

The canary is a constant-color image with a fixed label, appended to CIFAR-10 so
Poisson sampling puts it in the batch with rate q (this is B_t). At every logical
step we log the two quantities the reduction to Model 1 needs:

  Assumption 2 (exact saturation):  a_t = ||clip_C(grad canary)|| == C, i.e. ||g_t|| >= C.
  Assumption 1 (background suppression):  h_{t,z} = <u_t, clip_C(grad z)> == 0,
      reported as the batch sums sum_h and sum_h2 (and their 1/q-scaled estimates
      of the sums over the full dataset that Proposition 3 writes).

u_t is the canary's own unit direction at theta_{t-1}, recomputed every step --
Proposition 4 allows it to move arbitrarily, so we do not require it to be stable.

Usage:
  python scripts/canary_saturation_run.py --epsilon 8 --target-steps 200 \
      --canary-color 1,0,0 --canary-label 6 --out canary_sat.csv
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
from opacus.utils.batch_memory_manager import BatchMemoryManager
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(os.path.join(project_dir, 'src'))

from network_arch import WideResNet  # noqa: E402
from train import _augment_per_image  # same RandomCrop(reflect)+flip as training  # noqa: E402
from utils import setup_logging  # noqa: E402


class ConstantCanary(torch.utils.data.Dataset):
    """Single constant-color image with a fixed label."""

    def __init__(self, rgb, label):
        self.img = torch.tensor(rgb, dtype=torch.float32).view(3, 1, 1).expand(3, 32, 32).contiguous()
        self.label = int(label)

    def __len__(self):
        return 1

    def __getitem__(self, i):
        return self.img.clone(), self.label


def canary_gradient(model, optimizer, canary_img, label, normalize, device, K):
    """Per-example gradient of the canary at the current iterate, aug-averaged like training.

    Hooks are disabled so this leaves grad_sample untouched; p.grad after a mean-reduced
    backward over K augs is exactly the per-example gradient the clipper would see.
    """
    model.disable_hooks()
    model.zero_grad(set_to_none=True)
    x = canary_img.unsqueeze(0).to(device)
    x = torch.stack([_augment_per_image(x)[0] for _ in range(K)]) if K > 1 else x
    x = normalize(x)
    y = torch.full((x.shape[0],), label, dtype=torch.long, device=device)
    nn.functional.cross_entropy(model(x), y, reduction='mean').backward()
    g = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in optimizer.params]
    model.zero_grad(set_to_none=True)
    model.enable_hooks()
    norm = math.sqrt(sum(float((t * t).sum()) for t in g))
    u = [t / norm for t in g] if norm > 0 else g
    return norm, u


def dirac_direction(optimizer, flat_idx):
    """Unit vector e_c over the trainable params, c given as a global flat index."""
    u, off = [], 0
    for p in optimizer.params:
        t = torch.zeros_like(p)
        if off <= flat_idx < off + p.numel():
            t.view(-1)[flat_idx - off] = 1.0
        off += p.numel()
        u.append(t)
    if not (0 <= flat_idx < off):
        raise ValueError(f"--dirac {flat_idx} out of range (total coords={off})")
    return u


def batch_projections(optimizer, u, C, canary_rows):
    """sum_z h_{t,z} and sum_z h^2_{t,z} over the background records in this physical batch."""
    dots = norms_sq = None
    for p, up in zip(optimizer.params, u):
        gs = p.grad_sample
        if gs is None:
            continue
        flat = gs.reshape(gs.shape[0], -1)
        d = flat @ up.reshape(-1)
        s = (flat * flat).sum(dim=1)
        dots = d if dots is None else dots + d
        norms_sq = s if norms_sq is None else norms_sq + s
    if dots is None:
        return 0.0, 0.0, 0
    scale = (C / norms_sq.clamp_min(1e-12).sqrt()).clamp_max(1.0)   # per-sample clip factor
    h = scale * dots
    if canary_rows is not None and canary_rows.any():
        h = h[~canary_rows]
    return float(h.sum()), float((h * h).sum()), int(h.numel())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--canary-color', default='1,0,0', help='RGB in [0,1], comma separated')
    ap.add_argument('--canary-label', type=int, default=6)
    ap.add_argument('--dirac', type=int, default=None, metavar='FLAT_IDX',
                    help='baseline arm: score along dirac e_c instead of an input canary. '
                         'A2 is exact by construction; this measures A1 for the dirac audit.')
    ap.add_argument('--epsilon', type=float, default=8.0)
    ap.add_argument('--delta', type=float, default=1e-5)
    ap.add_argument('--target-steps', type=int, default=200)
    ap.add_argument('--logical-batch-size', type=int, default=4096)
    ap.add_argument('--max-physical-batch-size', type=int, default=128)
    ap.add_argument('--aug-multiplicity', type=int, default=1)
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
    logger.info(f"device={device} seed={seed} canary=const{rgb} label={args.canary_label} C={C}")

    base = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    canary_ds = ConstantCanary(rgb, args.canary_label)
    canary_img = canary_ds.img
    # dirac arm: no input canary in the data, direction is a fixed coordinate.
    train_ds = base if args.dirac is not None else torch.utils.data.ConcatDataset([base, canary_ds])
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.logical_batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    logger.info(f"dataset: {len(base)} CIFAR + 1 canary = {len(train_ds)}")

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

    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    canary_flat = canary_img.reshape(-1).to(device)

    rows = []
    step = 0
    u = None
    acc_h = acc_h2 = 0.0
    acc_n = 0
    model.train()
    done = False
    for epoch in range(1, epochs_for_priv + 1):
        if done:
            break
        with BatchMemoryManager(data_loader=train_loader, max_physical_batch_size=args.max_physical_batch_size, optimizer=optimizer) as loader:
            pbar = tqdm(loader, desc=f"epoch {epoch}", unit="batch")
            for images, labels in pbar:
                if u is None:   # start of a logical step: freeze u_t at theta_{t-1}
                    if args.dirac is not None:
                        u, g_norm = dirac_direction(optimizer, args.dirac), float('inf')
                    else:
                        g_norm, u = canary_gradient(model, optimizer, canary_img, args.canary_label, normalize, device, args.aug_multiplicity)
                images, labels = images.to(device), labels.to(device)
                canary_rows = (images.reshape(images.shape[0], -1) == canary_flat).all(dim=1)

                x = images
                if args.aug_multiplicity > 1:
                    x = torch.stack([_augment_per_image(images) for _ in range(args.aug_multiplicity)]).transpose(0, 1).reshape(-1, 3, 32, 32)
                    labels_k = labels.repeat_interleave(args.aug_multiplicity)
                else:
                    labels_k = labels
                x = normalize(x)

                optimizer.zero_grad()
                loss = nn.functional.cross_entropy(model(x), labels_k, reduction='mean')
                loss.backward()
                if args.aug_multiplicity > 1:
                    for p in model.parameters():
                        if getattr(p, 'grad_sample', None) is not None:
                            gs = p.grad_sample
                            p.grad_sample = gs.view(gs.shape[0] // args.aug_multiplicity, args.aug_multiplicity, *gs.shape[1:]).mean(dim=1)

                h, h2, n = batch_projections(optimizer, u, C, canary_rows)
                acc_h += h
                acc_h2 += h2
                acc_n += n

                optimizer.step()
                if getattr(optimizer, '_is_last_step_skipped', False):
                    continue   # partial logical batch: keep accumulating

                step += 1
                rows.append(dict(step=step, epoch=epoch, g_norm=g_norm, ratio=g_norm / C, saturated=int(g_norm >= C), sum_h=acc_h, sum_h2=acc_h2, n_bg=acc_n, mean_h=acc_h / max(1, acc_n), sum_h_D=acc_h / q, sum_h2_D=acc_h2 / q, loss=loss.item()))
                if step % 10 == 0 or step == 1:
                    logger.info(f"step {step:5d} ||g||/C={g_norm / C:8.3f} sum_h={acc_h:+.4f} mean_h={acc_h / max(1, acc_n):+.5f} sum_h2={acc_h2:.4f} n_bg={acc_n} loss={loss.item():.3f}")
                u = None
                acc_h = acc_h2 = 0.0
                acc_n = 0
                if step >= args.target_steps:
                    done = True
                    break

    ratios = np.array([r['ratio'] for r in rows])
    sum_h = np.array([r['sum_h_D'] for r in rows])
    sum_h2 = np.array([r['sum_h2_D'] for r in rows])
    mean_h = np.array([r['mean_h'] for r in rows])
    logger.info("=" * 70)
    if args.dirac is not None:
        logger.info(f"A2 saturation: exact by construction (dirac arm, coord {args.dirac}), {len(rows)} steps")
    else:
        logger.info(f"A2 saturation: min ||g||/C = {ratios.min():.3f} over {len(ratios)} steps; "
                    f"frac >= C = {np.mean(ratios >= 1.0):.4f}; argmin at step {rows[int(ratios.argmin())]['step']}")
    logger.info(f"A1 per-record alignment: mean_z h = {mean_h.mean():+.6f} (|clipped grad| <= C = {C})")
    logger.info(f"A1 background: sum_z h   (est over D) mean={sum_h.mean():+.4f} max|.|={np.abs(sum_h).max():.4f}   -> drift q*sum_h={q * sum_h.mean():+.6f} vs canary q*C={q * C:.6f}")
    logger.info(f"A1 background: sum_z h^2 (est over D) mean={sum_h2.mean():.4f}         -> var term q(1-q)*sum_h2={q * (1 - q) * sum_h2.mean():.6f} vs canary q(1-q)C^2={q * (1 - q) * C * C:.6f} vs tau^2={tau ** 2:.6f}")
    logger.info(f"A2 {'HOLDS' if ratios.min() >= 1.0 else 'FAILS'} on this run.")
    logger.info(f"A1 {'HOLDS (drift and var terms negligible vs tau^2)' if q * abs(sum_h.mean()) < 0.05 * q * C and q * (1 - q) * sum_h2.mean() < 0.05 * tau ** 2 else 'FAILS: background projections are not negligible'}")

    with open(args.out, 'w') as f:
        f.write(','.join(rows[0].keys()) + '\n')
        for r in rows:
            f.write(','.join(str(v) for v in r.values()) + '\n')
    meta = dict(seed=seed, sigma=sigma, q=q, tau=tau, C=C, canary_color=rgb, canary_label=args.canary_label, epsilon=args.epsilon, delta=args.delta, steps=len(rows), aug_multiplicity=args.aug_multiplicity, min_ratio=float(ratios.min()), frac_saturated=float(np.mean(ratios >= 1.0)))
    with open(os.path.splitext(args.out)[0] + '_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    logger.info(f"wrote {args.out}")


if __name__ == '__main__':
    main()
