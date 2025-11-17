#!/usr/bin/env python3
"""
Train a configurable MLP on CIFAR-10 and, at configurable intervals, compute for each hidden layer:
  - Using a large evaluation batch (e.g., 1000 samples, separate from the train minibatch),
  - The correlation matrix of units for both pre-activation (post-linear) and post-activation (after nonlinearity),
  - Then the scalar statistic mean_{i!=j} [ c_ij^2 * (1 - c_ij^2) ],
and plot this statistic vs. training step, color-coded by layer index.

Usage (examples):
    python cifar_mlp_correlation.py --epochs 2 --hidden-sizes 512 512 --activation relu --batch-size 256 \
        --eval-batch-size 1000 --eval-interval 100 --lr 1e-3

Notes:
  - Requires: torch, torchvision, matplotlib.
  - Uses CUDA if available.
"""

import argparse
import math
import os
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

import torchvision
import torchvision.transforms as T

import matplotlib
# matplotlib.use("Agg")  # Safe for headless environments
import matplotlib.pyplot as plt


# -----------------------------
# Configuration
# -----------------------------

@dataclass
class Config:
    data_root: str = "./data"
    epochs: int = 2
    batch_size: int = 256
    eval_batch_size: int = 1000           # large batch for correlation computation
    eval_interval: int = 100              # steps between correlation evaluations
    hidden_sizes: List[int] = field(default_factory=lambda: [512, 512])
    activation: str = "relu"              # relu | gelu | tanh | sigmoid | leaky_relu | elu | swish
    lr: float = 1e-3
    weight_decay: float = 0.0
    seed: int = 42
    num_workers: int = 2
    limit_train_batches: int = 0          # 0 means no limit (use all)
    limit_eval_samples: int = 0           # 0 means take eval_batch_size (or whatever available)
    save_dir: str = "./runs_cifar_mlp_corr"
    plot_prefix: str = "corrstat"         # prefix for saved plot files
    grayscale: bool = False               # if True, converts images to 1-channel


def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=False)
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01, inplace=False)
    if name == "elu":
        return nn.ELU(inplace=False)
    if name == "swish":
        # swish = x * sigmoid(x)
        class Swish(nn.Module):
            def forward(self, x):
                return x * torch.sigmoid(x)
        return Swish()
    raise ValueError(f"Unknown activation: {name}")


# -----------------------------
# Model
# -----------------------------

class MLP(nn.Module):
    """MLP that returns pre-activations and post-activations for each hidden layer."""
    def __init__(self, input_dim: int, hidden_sizes: List[int], num_classes: int, activation: nn.Module):
        super().__init__()
        self.layers = nn.ModuleList()
        self.activ = activation

        prev = input_dim
        for h in hidden_sizes:
            self.layers.append(nn.Linear(prev, h))
            prev = h
        self.out = nn.Linear(prev, num_classes)

    def forward(self, x, return_hidden: bool = True):
        """
        Args:
            x: shape [B, input_dim]
        Returns:
            logits, (pre_list, post_list) if return_hidden else logits
            - pre_list[k]: pre-activation (post-linear) of hidden layer k, shape [B, H_k]
            - post_list[k]: post-activation (after nonlinearity) of hidden layer k, shape [B, H_k]
        """
        pre_list = []
        post_list = []

        h = x
        for lin in self.layers:
            pre = lin(h)               # pre-activation (post-linear)
            post = self.activ(pre)     # post-activation (after nonlinearity)
            pre_list.append(pre)
            post_list.append(post)
            h = post
        logits = self.out(h)
        if return_hidden:
            return logits, (pre_list, post_list)
        return logits


# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def corr_matrix(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute correlation matrix over features (units) given batch dimension.
    x: [B, N] - B samples, N units.
    Returns: [N, N] correlation matrix.
    """
    B, N = x.shape
    x = x - x.mean(dim=0, keepdim=True)
    cov = (x.t() @ x) / max(B - 1, 1)
    var = cov.diag().clamp_min(eps)
    std = var.sqrt()
    denom = std[:, None] * std[None, :]
    corr = cov / denom.clamp_min(eps)
    corr = corr.clamp(min=-1.0, max=1.0)
    return corr


@torch.no_grad()
def corr_stat_offdiag(x: torch.Tensor) -> float:
    """
    Compute mean of c_ij^2 * (1 - c_ij^2) over off-diagonal entries.
    x: [B, N]
    """
    C = corr_matrix(x)  # [N, N]
    N = C.shape[0]
    off = ~torch.eye(N, dtype=torch.bool, device=C.device)
    c = C[off]
    val = (c**2 * (1 - c**2)).mean().item() if c.numel() > 0 else float("nan")
    return val


def prepare_data(cfg: Config):
    tfms = []
    if cfg.grayscale:
        tfms.append(T.Grayscale(num_output_channels=1))
    tfms += [T.ToTensor()]
    transform = T.Compose(tfms)

    trainset = torchvision.datasets.CIFAR10(root=cfg.data_root, train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=cfg.data_root, train=False, download=True, transform=transform)

    if cfg.limit_train_batches > 0:
        # Estimate number of samples to match limit of steps
        est_samples = cfg.limit_train_batches * cfg.batch_size
        trainset = Subset(trainset, list(range(min(est_samples, len(trainset)))))

    trainloader = DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True,
                             num_workers=cfg.num_workers, pin_memory=True)

    testloader = DataLoader(testset, batch_size=cfg.eval_batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)
    return trainloader, testloader


def flatten_images(batch: torch.Tensor) -> torch.Tensor:
    # batch: [B, C, H, W] -> [B, C*H*W]
    return batch.view(batch.size(0), -1)


def make_eval_batch(testloader: DataLoader, cfg: Config, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Grab a single large batch from the test set for correlation analysis.
    Returns: (images_flat, labels) on the specified device.
    """
    xs, ys = [], []
    target_n = cfg.limit_eval_samples if cfg.limit_eval_samples > 0 else cfg.eval_batch_size
    n_collected = 0
    for x, y in testloader:
        xs.append(x)
        ys.append(y)
        n_collected += x.size(0)
        if n_collected >= target_n:
            break
    X = torch.cat(xs, dim=0)[:target_n]
    Y = torch.cat(ys, dim=0)[:target_n]
    X = flatten_images(X).to(device, non_blocking=True)
    Y = Y.to(device, non_blocking=True)
    return X, Y


def build_model(cfg: Config, input_dim: int, num_classes: int, device: torch.device) -> MLP:
    model = MLP(input_dim=input_dim, hidden_sizes=cfg.hidden_sizes, num_classes=num_classes,
                activation=get_activation(cfg.activation)).to(device)
    return model


def train_and_analyze(cfg: Config):
    os.makedirs(cfg.save_dir, exist_ok=True)

    set_seed(cfg.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available(): # For Apple Silicon GPUs
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    # print(f"Using device: {device}")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    trainloader, testloader = prepare_data(cfg)
    # Determine input dimension from one batch
    example_batch, _ = next(iter(trainloader))
    C, H, W = example_batch.shape[1:]
    input_dim = C * H * W
    num_classes = 10

    model = build_model(cfg, input_dim=input_dim, num_classes=num_classes, device=device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Fixed evaluation batch for correlation stats
    X_eval, _ = make_eval_batch(testloader, cfg, device)

    # History containers
    steps = []
    pre_stats_history: Dict[int, List[float]] = {i: [] for i in range(len(cfg.hidden_sizes))}
    post_stats_history: Dict[int, List[float]] = {i: [] for i in range(len(cfg.hidden_sizes))}

    global_step = 0
    model.train()
    for epoch in range(cfg.epochs):
        for i, (x, y) in enumerate(trainloader):
            x = flatten_images(x.to(device, non_blocking=True))
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(x, return_hidden=True)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()

            if (global_step % cfg.eval_interval) == 0:
                with torch.no_grad():
                    # Forward the large eval batch and collect hidden states
                    _, (pre_list, post_list) = model(X_eval, return_hidden=True)
                    # Compute per-layer correlation stats
                    for li, pre in enumerate(pre_list):
                        stat_pre = corr_stat_offdiag(pre)
                        pre_stats_history[li].append(stat_pre)
                    for li, post in enumerate(post_list):
                        stat_post = corr_stat_offdiag(post)
                        post_stats_history[li].append(stat_post)
                    steps.append(global_step)
                print(f"[step {global_step}] loss={loss.item():.4f}")

            global_step += 1

            # Optional early stop for debugging small runs
            if cfg.limit_train_batches > 0 and global_step >= cfg.limit_train_batches:
                break
        if cfg.limit_train_batches > 0 and global_step >= cfg.limit_train_batches:
            break

    # --------------- Plotting ---------------
    # One plot for pre-activations (post-linear), one for post-activations (after nonlinearity).
    def plot_history(history: Dict[int, List[float]], steps: List[int], title: str, filename: str):
        plt.figure(figsize=(8, 5))
        num_layers = len(history)
        for li in range(num_layers):
            y = history[li]
            # steps and y should align; if last step missing due to edge cases, trim to min len
            L = min(len(steps), len(y))
            plt.plot(steps[:L], y[:L], label=f"Layer {li}")
        plt.xlabel("Training step t")
        plt.ylabel(r"mean offdiag $c_{ij}^2 (1 - c_{ij}^2)$")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        out_path = os.path.join(cfg.save_dir, filename)
        plt.savefig(out_path, dpi=150)
        print(f"Saved plot: {out_path}")

    plot_history(pre_stats_history, steps, "Pre-activation (post-linear) correlation statistic", f"{cfg.plot_prefix}_pre.png")
    plot_history(post_stats_history, steps, "Post-activation correlation statistic", f"{cfg.plot_prefix}_post.png")

    # Also save a small report of the final values
    report_path = os.path.join(cfg.save_dir, "final_stats.txt")
    with open(report_path, "w") as f:
        f.write("Final correlation statistic per layer (last checkpoint):\n")
        for li in range(len(cfg.hidden_sizes)):
            pre_val = pre_stats_history[li][-1] if pre_stats_history[li] else float("nan")
            post_val = post_stats_history[li][-1] if post_stats_history[li] else float("nan")
            f.write(f"Layer {li}: pre={pre_val:.6f}, post={post_val:.6f}\n")
    print(f"Wrote summary: {report_path}")

    return {
        "pre_plot": os.path.join(cfg.save_dir, f"{cfg.plot_prefix}_pre.png"),
        "post_plot": os.path.join(cfg.save_dir, f"{cfg.plot_prefix}_post.png"),
        "report": report_path,
    }


def parse_args_to_config() -> Config:
    p = argparse.ArgumentParser(description="CIFAR-10 MLP correlation experiment")
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--eval-batch-size", type=int, default=1000)
    p.add_argument("--eval-interval", type=int, default=100)
    p.add_argument("--hidden-sizes", type=int, nargs="+", default=[512, 512])
    p.add_argument("--activation", type=str, default="relu",
                   choices=["relu", "gelu", "tanh", "sigmoid", "leaky_relu", "elu", "swish"])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--limit-train-batches", type=int, default=0, help="Optional cap on number of train steps (0=none).")
    p.add_argument("--limit-eval-samples", type=int, default=0, help="Optional cap on eval batch size (0=use eval_batch_size).")
    p.add_argument("--save-dir", type=str, default="./runs_cifar_mlp_corr")
    p.add_argument("--plot-prefix", type=str, default="corrstat")
    p.add_argument("--grayscale", action="store_true", help="Convert images to grayscale before flattening.")
    args = p.parse_args()

    cfg = Config(
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        eval_interval=args.eval_interval,
        hidden_sizes=args.hidden_sizes,
        activation=args.activation,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        num_workers=args.num_workers,
        limit_train_batches=args.limit_train_batches,
        limit_eval_samples=args.limit_eval_samples,
        save_dir=args.save_dir,
        plot_prefix=args.plot_prefix,
        grayscale=args.grayscale,
    )
    return cfg


def main():
    cfg = parse_args_to_config()
    train_and_analyze(cfg)


if __name__ == "__main__":
    main()
