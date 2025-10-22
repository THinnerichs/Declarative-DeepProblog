# networks/network.py
from typing import Optional
from torch import nn
import torch
import torch.nn.functional as F


# ---------------- Encoder (parametric emb_dim = 12 by default) ----------------
class SymbolEncoder(nn.Module):
    def __init__(self, emb_dim: int = 12):
        super().__init__()
        self.emb_dim = emb_dim
        self.convolutions = nn.Sequential(
            nn.Conv2d(1, 6, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),              # 45 -> 22
            nn.Conv2d(6, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),              # 22 -> 11
            nn.Dropout2d(0.4),
        )
        self.mlp = nn.Sequential(
            nn.Linear(16 * 11 * 11, emb_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.convolutions(x)          # [B,16,11,11]
        x = torch.flatten(x, 1)           # [B,16*11*11]
        x = self.mlp(x)                   # [B,emb_dim]
        return x


# ---------------- Decoder (to exactly 45x45, outputs in [-0.5, 0.5]) ----------------
class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape  # e.g., (-1, 32, 6, 6)

    def forward(self, x):
        return x.view(*self.shape)


class SymbolDecoder(nn.Module):
    """
    ConvTranspose2d-based decoder for HWF images (1x45x45).
    Latent z -> fc -> (32,6,6) -> deconv x3 -> (1,45,45) -> tanh scaled to [-0.5, 0.5].

    Geometry:
      6x6 --(k3,s2,p1,op=1)--> 12x12
      12x12 --(k3,s2,p1,op=0)--> 23x23
      23x23 --(k3,s2,p1,op=0)--> 45x45
    """
    def __init__(self, emb_dim: int = 12):
        super().__init__()
        self.emb_dim = emb_dim
        self.fc = nn.Linear(emb_dim, 32 * 6 * 6)

        self.Decoder = nn.Sequential(
            nn.Linear(emb_dim, 32 * 6 * 6),
            Reshape((-1, 32, 6, 6)),
            nn.ReLU(inplace=True),

            # 6 -> 12  (output_padding=1)
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),

            # 12 -> 23 (output_padding=0)
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.ReLU(inplace=True),

            # 23 -> 45 (output_padding=0)
            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=1, output_padding=0),

            nn.Tanh(),   # -> [-1,1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        img = self.Decoder(z)          # [B,1,45,45] in [-1,1]
        img = 0.5 * img                # scale to [-0.5, 0.5] to match your data
        return img



class ProtoSymbol(nn.Module):
    """
    Probabilistic prototypes with explicit .prototypes (means).
    - prototypes: [C,D] = μ_c
    - logvar:     [C,D] = log σ_c^2
    - latent score: Gaussian log-likelihood of z under class c
    - recon score: MC average over samples z~N(μ_c, σ_c^2 I), decoded to image space
    Outputs probabilities over classes.
    """
    def __init__(
        self,
        encoder: nn.Module,
        n_classes: int,
        emb_dim: int = 12,
        decoder: Optional[nn.Module] = None,
        n_mc_protos: int = 1,
        epsilon_gate: float = 0.1,
        init_gate_logits: tuple = (0.0, 0.0),
        recon_weight: float = 1.0, 
    ):
        super().__init__()
        self.encoder = encoder
        self.emb_dim = emb_dim
        self.n_classes = n_classes
        self.decoder = decoder
        self.n_mc = n_mc_protos
        self.epsilon_gate = epsilon_gate
        self.recon_scale = recon_weight 

        # explicit prototypes + logvar (unchanged)
        self.prototypes = nn.Parameter(torch.randn(n_classes, emb_dim) * 0.02)
        self.logvar     = nn.Parameter(torch.full((n_classes, emb_dim), -1.0))

        self.gate_logits = nn.Parameter(torch.tensor(init_gate_logits, dtype=torch.float))


    # Gaussian log-likelihood of z under each class (averaged over dims)
    def _latent_loglik(self, z: torch.Tensor) -> torch.Tensor:
        inv_var = torch.exp(-self.logvar)                           # [C,D]
        diff = z.unsqueeze(1) - self.prototypes.unsqueeze(0)        # [B,C,D]
        ll = -0.5 * ((diff ** 2) * inv_var + self.logvar.unsqueeze(0)).mean(dim=-1)  # [B,C]
        return ll

    # Sample prototype latents for each class: [K,C,D] -> decode -> [K,C,1,45,45]
    def _decode_proto_samples(self) -> Optional[torch.Tensor]:
        if self.decoder is None:
            return None
        K = self.n_mc
        std = torch.exp(0.5 * self.logvar)                          # [C,D]
        eps = torch.randn(K, self.n_classes, self.emb_dim, device=self.prototypes.device)
        z = self.prototypes.unsqueeze(0) + eps * std.unsqueeze(0)   # [K,C,D]
        z = z.view(K * self.n_classes, self.emb_dim)
        imgs = self.decoder(z)                                       # [K*C,1,45,45] in [-0.5,0.5]
        return imgs.view(K, self.n_classes, 1, 45, 45)

    @staticmethod
    def _standardize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        m, s = x.mean(), x.std().clamp_min(eps)
        return (x - m) / s

    def _mix_weights(self):
        w = F.softmax(self.gate_logits, dim=0)    # [2]
        eps = self.epsilon_gate
        w_lat = eps + (1 - 2 * eps) * w[0]
        w_rec = 1.0 - w_lat
        return w_lat, w_rec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # latent term
        z = self.encoder(x)               # [B,D]
        s_lat = self._standardize(self._latent_loglik(z))  # [B,C]

        proto_imgs = self._decode_proto_samples()
        if proto_imgs is not None:
            x_ = x.unsqueeze(0).unsqueeze(2)                 # [1,B,1,1,45,45]
            diff = x_ - proto_imgs.unsqueeze(1)              # [K,B,C,1,45,45]
            mse = (diff ** 2).mean(dim=(0, 3, 4, 5))         # [B,C]
            s_rec = self._standardize(-mse)
        else:
            s_rec = torch.zeros_like(s_lat)

        w_lat, w_rec = self._mix_weights()
        logits = w_lat * s_lat + w_rec * (self.recon_scale * s_rec)  # <--- UPDATED
        return F.softmax(logits, dim=-1)

    # convenience: decode mean prototypes once (no sampling)
    @torch.no_grad()
    def decode_mean_prototypes(self) -> torch.Tensor:
        if self.decoder is None:
            raise RuntimeError("ProtoSymbol.decoder is None; cannot decode prototypes.")
        return self.decoder(self.prototypes)  # [C,1,45,45] in [-0.5,0.5]
