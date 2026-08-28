# networks/network.py
from typing import Optional
from torch import nn
import torch
import math
import torch.nn.functional as F

IMG_SIZE = 45


# ---------- encoder for 45x45 ----------
class SymbolEncoder(nn.Module):
    """
    45x45 grayscale -> latent vector (emb_dim).
    Path: 45 -> 15 -> 5 feature map, then linear.
    """
    def __init__(self, emb_dim: int = 12, ch1=16, ch2=32, ch3=32, dropout=0.2):
        super().__init__()
        self.emb_dim = emb_dim
        self.conv = nn.Sequential(
            nn.Conv2d(1, ch1, 3, padding=1), nn.ELU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3),          # 45 -> 15

            nn.Conv2d(ch1, ch2, 3, padding=1), nn.ELU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3),          # 15 -> 5

            nn.Conv2d(ch2, ch3, 3, padding=1), nn.ELU(inplace=True),
            nn.Dropout2d(dropout),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ch3 * 5 * 5, emb_dim),                # <-- 5x5 here
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)                  # [B, ch3, 5, 5]
        z = self.fc(x)                    # [B, emb_dim]
        return z


# -------------------- decoder for 45x45 (logits) --------------------
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=True)
        self.act   = nn.ELU(inplace=True)
    def forward(self, x):
        y = self.act(self.conv1(x))
        y = self.conv2(y)
        return self.act(x + y)

class SymbolDecoder(nn.Module):
    """
    z -> (c0,5,5) -> deconv x2 -> (1,45,45) logits.
    5->15: (k=3, s=3, p=0)
    15->45: (k=3, s=3, p=0)
    """
    def __init__(self, emb_dim: int = 12, c0=32, c1=24, c2=16):
        super().__init__()
        self.fc  = nn.Linear(emb_dim, c0 * 5 * 5)

        self.d1  = nn.ConvTranspose2d(c0, c1, kernel_size=3, stride=3, padding=0)  # 5 -> 15
        self.r1  = ResBlock(c1)

        self.d2  = nn.ConvTranspose2d(c1, c2, kernel_size=3, stride=3, padding=0)  # 15 -> 45
        self.r2  = ResBlock(c2)

        self.out = nn.Conv2d(c2, 1, kernel_size=1)  # logits

        # init
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity='relu'); nn.init.zeros_(self.fc.bias)
        for m in [self.d1, self.d2, self.out]:
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).view(z.size(0), -1, 5, 5)           # (B,c0,5,5)
        x = F.elu(self.d1(x), inplace=True); x = self.r1(x)  # (B,c1,15,15)
        x = F.elu(self.d2(x), inplace=True); x = self.r2(x)  # (B,c2,45,45)
        x = self.out(x)  # (B,1,45,45)

        x = x.tanh() * 0.5

        return x

class ProtoSymbol(nn.Module):
    """
    Prototype classifier for DeepStochlog.

    For each input image x:

      1. Encode x -> z (embedding μ(x)).
      2. For each class c, compute a latent Gaussian "likelihood":
            log_latent_c = -β * 0.5 * Σ_d ((μ_d - e_cd)^2 / σ^2)
      3. Decode each prototype e_c to an image and compute MSE to x in [0,1]:
            mse_c = MSE(ink(x), ink(dec(e_c)))
         then
            log_rec_c = -α * mse_c
      4. Compute a diversity prior from decoded prototype pixel variance:
            var_c = Var(ink(dec(e_c)))
            log_div_c = γ * (var_c - τ)
         (flat / low-variance prototypes get negative log_div)
      5. Combine:
            log_score_c = log_latent_c + log_rec_c + w_div * log_div_c
         and softmax over c to get p(c | x).

    This keeps things simple, avoids over-normalizing, and lets the diversity
    prior have a consistent effect across all inputs.
    """

    def __init__(
        self,
        encoder: nn.Module,
        n_classes: int,
        emb_dim: int = 12,
        decoder: Optional[nn.Module] = None,
        recon_weight: float = 1.0,      # α: weight on MSE(x, dec(proto))
        n_mc_protos: int = 1,           # kept for compatibility (used only in decode_proto_samples)
        # kept for compatibility, but we use fixed variances here:
        bound_variances: bool = True,
        logvar_min: float = -4.0,
        logvar_max: float =  2.0,
        #
        prior_std: float = 1.50,        # fixed std for p(z | c)
        sample_std: float = 0.30,       # std for sampling around prototypes
        beta_kl: float = 1.0,           # β: scale of latent distance term
        diversity_w: float = 0.0,       # w_div: strength of diversity prior
        min_ink_coverage: float = 0.02, # τ: variance threshold in diversity prior
        coverage_w: float = 5.0,        # γ: scale for diversity prior
    ):
        super().__init__()
        self.encoder = encoder
        self.n_classes = n_classes
        self.emb_dim = emb_dim
        self.decoder = decoder
        if self.decoder is None:
            raise RuntimeError("ProtoSymbol requires a decoder for image-based terms.")

        self.recon_weight = float(recon_weight)
        self.n_mc = int(n_mc_protos)  # not used in forward, only in sampling

        self.beta_kl = float(beta_kl)
        self.diversity_w = float(diversity_w)
        self.min_ink_coverage = float(min_ink_coverage)  # used as τ (variance threshold)
        self.coverage_w = float(coverage_w)              # γ (scale in diversity prior)

        # Prototype means (trainable): e_c
        self.prototypes = nn.Parameter(torch.randn(n_classes, emb_dim) * 0.02)

        # Fixed stds (non-trainable) in latent space
        self.register_buffer(
            "prior_logvar",
            torch.full((1, emb_dim), math.log(prior_std ** 2), dtype=torch.float32),
        )
        self.register_buffer(
            "sample_std",
            torch.full((1, emb_dim), float(sample_std), dtype=torch.float32),
        )

        # Track usage (not directly used in forward, but we keep updating it)
        self.register_buffer("usage_ema", torch.full((n_classes,), 1.0 / n_classes))
        self.usage_momentum = 0.99

    # -------- helpers --------

    @staticmethod
    def _to_ink01(img: torch.Tensor) -> torch.Tensor:
        """
        Map images in [-0.5, 0.5] (white≈+0.5, ink≈-0.5) to an ink-positive [0,1] scale.
        """
        return (0.5 - img).clamp(0.0, 1.0)

    @torch.no_grad()
    def _decode_proto_means(self) -> torch.Tensor:
        """
        Decode the MEAN prototypes.

        Returns:
            imgs: [C,1,H,W] in the same range as decoder output (assumed [-0.5,0.5]).
        """
        imgs = self.decoder(self.prototypes)  # [C,1,H,W]
        return imgs

    # -------- core forward --------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,1,H,W] in [-0.5,0.5]
        returns:
            probs: [B,C] class distribution
        """
        B, _, H, W = x.shape
        C, D = self.n_classes, self.emb_dim
        device = x.device
        dtype = x.dtype
        eps = 1e-12

        # --------------------------------------------------------
        # 1) Latent term: Gaussian "likelihood" under each proto
        # --------------------------------------------------------
        z = self.encoder(x)  # [B,D] or (mu, ...)
        if isinstance(z, (tuple, list)):
            mu = z[0]
        else:
            mu = z  # [B,D]

        prior_var = self.prior_logvar.exp()                # [1,D]
        diff = mu.unsqueeze(1) - self.prototypes.unsqueeze(0)  # [B,C,D]
        energy = 0.5 * (diff.pow(2) / prior_var).sum(dim=-1)   # [B,C]

        # log_latent_c ∝ -β * energy_c
        log_latent = -self.beta_kl * energy                 # [B,C]

        # --------------------------------------------------------
        # 2) Reconstruction term: MSE in ink space ∈ [0,1]
        # --------------------------------------------------------
        proto_imgs = self._decode_proto_means()             # [C,1,H,W], [-0.5,0.5]

        x01 = self._to_ink01(x)                             # [B,1,H,W]
        p01 = self._to_ink01(proto_imgs)                    # [C,1,H,W]

        x_bc = x01.unsqueeze(1).expand(-1, C, -1, -1, -1)   # [B,C,1,H,W]
        p_bc = p01.unsqueeze(0).expand(B, -1, -1, -1, -1)   # [B,C,1,H,W]

        mse = (x_bc - p_bc).pow(2).mean(dim=(2, 3, 4))      # [B,C], theoretically ∈ [0,1]
        mse = mse.clamp(0.0, 1.0)

        # log_rec_c ∝ -α * mse_c
        log_rec = -self.recon_weight * mse                  # [B,C]

        # --------------------------------------------------------
        # 3) Diversity prior: encourage non-flat prototypes
        # --------------------------------------------------------
        proto_flat = p01.view(C, -1)                        # [C, H*W] in [0,1]
        pixel_var = proto_flat.var(dim=1)                   # [C]

        tau = float(self.min_ink_coverage)                  # variance threshold
        gamma = float(self.coverage_w)                      # scale

        # log_div_c = γ * (var_c - τ)
        log_div = gamma * (pixel_var - tau)                 # [C]

        if self.diversity_w <= 0.0:
            # no diversity prior
            log_div = torch.zeros_like(log_div)
        else:
            # scale overall strength
            log_div = self.diversity_w * log_div            # [C]

        log_div = log_div.to(device=device, dtype=dtype).unsqueeze(0)  # [1,C]

        # --------------------------------------------------------
        # 4) Combine in log-space and softmax
        # --------------------------------------------------------
        # log_score_c(x) = log_latent_c + log_rec_c + log_div_c
        log_score = log_latent + log_rec + log_div          # [B,C]

        # Numerical stability: subtract per-sample max
        log_score = log_score - log_score.max(dim=1, keepdim=True)[0]
        score = torch.exp(log_score)                        # [B,C]
        probs = score / (score.sum(dim=1, keepdim=True) + eps)

        # --------------------------------------------------------
        # 5) Update usage_ema (just for bookkeeping / inspection)
        # --------------------------------------------------------
        with torch.no_grad():
            m = float(self.usage_momentum)
            self.usage_ema.mul_(m).add_((1.0 - m) * probs.mean(dim=0))

        return probs

    # -------- sampling utility for grids / inspection --------

    @torch.no_grad()
    def decode_proto_samples(self, n_per_class: int = 1) -> torch.Tensor:
        """
        Sample around each prototype in latent space and decode.

        Returns:
            imgs: [C * n_per_class, 1, H, W] in [-0.5, 0.5].
        """
        if self.decoder is None:
            raise RuntimeError("decoder=None; cannot decode.")
        C, D = self.n_classes, self.emb_dim
        e = self.prototypes.unsqueeze(1).expand(C, n_per_class, D)  # [C,n_per,D]
        eps = torch.randn_like(e)
        z = e + eps * self.sample_std                               # [C,n_per,D]
        z = z.reshape(C * n_per_class, D)
        imgs = self.decoder(z)                                      # [C*n_per,1,H,W]
        return imgs

