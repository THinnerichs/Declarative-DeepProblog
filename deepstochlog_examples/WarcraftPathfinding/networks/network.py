from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepstochlog.network import Network, NetworkStore
from deepstochlog.term import Term

COST_BINS: Sequence[int] = (0, 1, 4)

class TileEncoder(nn.Module):
    """
    (C, H, W) tile -> latent vector z in R^D.
    Works for H,W >= 4 thanks to global pooling.
    """

    def __init__(self, in_ch: int, latent_dim: int = 16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),          # e.g. 8x8 -> 4x4

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),  # -> (128,1,1)
        )
        self.fc = nn.Linear(128, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x).flatten(1)   # (B,128)
        z = self.fc(h)                # (B,D)
        return z


class TileDecoder(nn.Module):
    """
    latent z -> (C,8,8) tile in [0,1].
    """

    def __init__(self, latent_dim: int, out_ch: int, tile_h: int = 8, tile_w: int = 8):
        super().__init__()
        self.tile_h = tile_h
        self.tile_w = tile_w
        self.fc = nn.Linear(latent_dim, 16 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 16, 4, stride=2, padding=1),  # (16,8,8)
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_ch, 3, padding=1),
            nn.Sigmoid(),                                       # [0,1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)                            # (B,16*4*4)
        h = h.view(z.size(0), 16, 4, 4)           # (B,16,4,4)
        x_hat = self.deconv(h)                    # (B,C,8,8)
        if x_hat.shape[-2:] != (self.tile_h, self.tile_w):
            x_hat = F.interpolate(
                x_hat,
                size=(self.tile_h, self.tile_w),
                mode="bilinear",
                align_corners=False,
            )
        return x_hat

class CostPrototypeNet(nn.Module):
    """
    Prototype-based tile classifier used inside DeepStochlog.
    """

    def __init__(
        self,
        in_ch: int,
        latent_dim: int = 16,
        tile_h: int = 8,
        tile_w: int = 8,
        cost_bins: Iterable[int] = COST_BINS,
    ):
        super().__init__()
        self.encoder = TileEncoder(in_ch, latent_dim)
        self.decoder = TileDecoder(latent_dim, in_ch, tile_h, tile_w)

        self.cost_bins = list(cost_bins)
        self.n_classes = len(self.cost_bins)
        self.latent_dim = latent_dim

        # Unlearnable latent prototypes: one per cost bin
        mu = torch.randn(self.n_classes, latent_dim)
        log_sigma = torch.zeros(self.n_classes, latent_dim)
        self.register_buffer("proto_mu", mu)          # (K,D)
        self.register_buffer("proto_log_sigma", log_sigma)  # (K,D)

    # ---- latent Gaussian log-likelihood per prototype ----
    def _prototype_log_probs(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B,D)
        returns log_probs: (B,K)  (up to an additive constant)
        """
        K = self.n_classes
        mu = self.proto_mu              # (K,D)
        log_sigma = self.proto_log_sigma
        sigma2 = torch.exp(2.0 * log_sigma)  # (K,D)

        z_exp = z.unsqueeze(1)                # (B,1,D)
        mu_exp = mu.unsqueeze(0)             # (1,K,D)
        sigma2_exp = sigma2.unsqueeze(0)     # (1,K,D)

        diff2 = (z_exp - mu_exp) ** 2        # (B,K,D)
        log_det = torch.sum(torch.log(2.0 * math.pi * sigma2_exp), dim=-1)  # (B,K)
        quad = torch.sum(diff2 / sigma2_exp, dim=-1)                        # (B,K)
        log_probs = -0.5 * (log_det + quad)                                 # (B,K)
        return log_probs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        device = x.device

        # --- encode ---
        z = self.encoder(x)                    # (B,D)

        # --- latent likelihood per prototype (normalized to [0,1], sum=1) ---
        log_probs = self._prototype_log_probs(z)            # (B,K)
        # Numerically stable softmax -> likelihood in [0,1], per-sample sums to 1
        lik = torch.softmax(log_probs, dim=-1)              # (B,K)

        # --- reconstruction error per prototype ---
        # Decode prototype means once
        z_proto = self.proto_mu.to(device)                  # (K,D)
        proto_decodes = self.decoder(z_proto)               # (K,C,H,W)

        x_exp = x.unsqueeze(1)                              # (B,1,C,H,W)
        proto_exp = proto_decodes.unsqueeze(0)              # (1,K,C,H,W)
        mse = (x_exp - proto_exp) ** 2
        mse = mse.mean(dim=(2, 3, 4))                       # (B,K)

        # Normalize MSE to [0,1] per example, then take similarity = 1 - m
        mse_min = mse.min(dim=1, keepdim=True).values
        mse_max = mse.max(dim=1, keepdim=True).values
        denom = (mse_max - mse_min).clamp(min=1e-6)
        mse_norm = (mse - mse_min) / denom                  # in [0,1]
        sim = 1.0 - mse_norm                                # in [0,1], 1 = best recon

        # --- combine: likelihood × similarity, then normalize to probabilities ---
        score = lik * sim                                   # (B,K), >= 0
        score = score + 1e-12
        probs = score / score.sum(dim=1, keepdim=True)      # (B,K)

        return probs

def build_networks(
    in_ch: int,
    latent_dim: int = 16,
    tile_h: int = 8,
    tile_w: int = 8,
    cost_bins: Sequence[int] = COST_BINS,
) -> NetworkStore:
    """
    Create DeepStochlog NetworkStore with a single prototype tile-cost network.

    DeepStochlog predicate: nn(cost_net, [X], Y, dom_cost).
    """
    model = CostPrototypeNet(
        in_ch=in_ch,
        latent_dim=latent_dim,
        tile_h=tile_h,
        tile_w=tile_w,
        cost_bins=cost_bins,
    )

    index_list = [Term(str(c)) for c in cost_bins]
    cost_network = Network("cost_net", model, index_list=index_list)

    return NetworkStore(cost_network)
