"""
ProtoVAE for the MathExpression HWF dataset using the official
train/val/test splits from expr_train.json / expr_val.json / expr_test.json.

Core idea:

- Encoder: q(z|x) = N(mu(x), diag(exp(lv(x)))).
- For each symbol class y, we have a Gaussian prior in latent space:
        p(z|y) = N(e_y, PRIOR_STD^2 * I)
  where e_y is a learned prototype for that class (digit or operator).

- Classification is likelihood-based:
        logits_c = -||mu(x) - e_c||^2 / (2 * PRIOR_STD^2)
        p(y|x) = softmax(logits_c)
  We train with cross-entropy on these logits.

- Reconstruction:
  We decode z_inst ~ q(z|x) to reconstruct the symbol image with
  Bernoulli pixels (BCE with logits).

- Generation (for sampling & generative accuracy):
  Given a class y, we sample
        z ~ N(e_y, GEN_SAMPLE_STD^2 I)
  and decode to an image.

- Generative accuracy:
  For each class y, sample N images from N(e_y, GEN_SAMPLE_STD^2 I),
  decode, and for each sample find the nearest neighbour (MSE) in a
  held-out dataset. Correct if NN label == y.
"""

import os
import math
import random
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

from mathexpression_data import (
    AbstractMathExprDataset,
    all_symbols_list,
)

# ============================================================
# Config
# ============================================================

SEED        = 0
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE    = 45          # image size (45x45 in the paper)
Z_DIM       = 32
HIDDEN_DIM  = 512         # MLP hidden size

TRAIN_SAMPLE_STD = 0.30   # std used in p(z|y) during training (for KL prior)
GEN_SAMPLE_STD   = 0.20   # std used when sampling from prototypes for generation

PRIOR_STD   = TRAIN_SAMPLE_STD

BATCH_SIZE  = 128
EPOCHS      = 15
LR          = 2e-3

# Loss weights
BETA_KL     = 0.1         # weight for KL term
ALPHA_CLS   = 1.0         # weight for likelihood-based classification term

OUT_DIR = "./out_vaehwf_splits"
os.makedirs(OUT_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Symbol vocabulary from the project (["UNK","1",...,"9","+","-","*","/"])
LABELS: List[str] = all_symbols_list
N_CLASSES = len(LABELS)

# ============================================================
# Utility: safe labels for filenames
# ============================================================

def safe_label_for_filename(lbl: str) -> str:
    """
    Convert symbol labels (+, -, *, /, etc.) into filename-safe strings.
    """
    mapping = {
        "+": "plus",
        "-": "minus",
        "*": "times",
        "/": "div",
        "=": "eq",
    }
    if lbl in mapping:
        return mapping[lbl]
    # replace any non-alnum with underscore
    out = "".join(ch if ch.isalnum() else "_" for ch in lbl)
    return out if out else "sym"

# ============================================================
# Symbol-level dataset from AbstractMathExprDataset
# ============================================================

class MathExprSymbolDataset(AbstractMathExprDataset, Dataset):
    """
    Turn the expression-level AbstractMathExprDataset into a symbol-level dataset
    of (image, label), where each symbol in each expression becomes one sample.

    Splits:
        split in {"train", "val", "test"} -> expr_{split}.json
    """

    def __init__(
        self,
        split: str = "train",
        num_samples=None,
        random_seed=None,
        expression_length=None,
        expression_max_length=None,
        allow_division=None,
    ):
        super().__init__(
            split=split,
            num_samples=num_samples,
            random_seed=random_seed,
            expression_length=expression_length,
            expression_max_length=expression_max_length,
            allow_division=allow_division,
        )

        # Flatten expression samples into symbol samples
        self.samples: List[Tuple[str, int]] = []
        for ex in self.dataset:
            img_seq = ex["image_sequence"]
            lab_seq = ex["label_sequence"]
            for img_id, lab in zip(img_seq, lab_seq):
                self.samples.append((img_id, lab))

        print(
            f"[MathExprSymbolDataset] split={split}, "
            f"expressions={len(self.dataset)}, symbols={len(self.samples)}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_id, label = self.samples[idx]
        # self.tensors[img_id] is [1,H,W] after transforms:
        #   ToTensor() + Normalize((0.5,), (1,))
        # so values are in [-0.5, 0.5] with background ~0.5 and ink ~-0.5.
        x_norm = self.tensors[img_id]

        # Map to [0,1] with 1 = ink, 0 = background:
        # background:  0.5  -> 0.5 - 0.5 = 0
        # ink:        -0.5 -> 0.5 - (-0.5) = 1
        x = 0.5 - x_norm
        x = x.clamp(0.0, 1.0)

        return x, int(label)


def make_loader(ds: Dataset, batch_size: int = BATCH_SIZE, shuffle: bool = True) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=True)

# ============================================================
# MLP Encoder / Decoder (size-agnostic)
# ============================================================

class EncoderMLP(nn.Module):
    def __init__(self, z: int = Z_DIM, img_size: int = IMG_SIZE, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.img_dim = img_size * img_size
        self.fc1 = nn.Linear(self.img_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, z)
        self.fc_lv = nn.Linear(hidden_dim, z)

    def forward(self, x: torch.Tensor):
        # x: [B, 1, H, W] in [0,1], 1=ink
        B = x.size(0)
        h = x.view(B, -1)
        h = F.relu(self.fc1(h))
        mu = self.fc_mu(h)
        lv = self.fc_lv(h)
        return mu, lv


class DecoderMLP(nn.Module):
    def __init__(self, z: int = Z_DIM, img_size: int = IMG_SIZE, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.img_dim = img_size * img_size
        self.img_size = img_size
        self.fc1 = nn.Linear(z, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, self.img_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(z))
        logits = self.fc_out(h)
        logits = logits.view(-1, 1, self.img_size, self.img_size)
        return logits  # [B,1,H,W], unnormalized ink logits

# ============================================================
# ProtoVAE (1 prototype per symbol class)
# ============================================================

class ProtoVAE(nn.Module):
    """
    Latent space with one prototype e_y per symbol class y.

    q(z|x) = N(mu, diag(exp(lv)))
    p(z|y) = N(e_y, PRIOR_STD^2 * I)
    """

    def __init__(
        self,
        z: int = Z_DIM,
        n_classes: int = N_CLASSES,
        sample_std: float = TRAIN_SAMPLE_STD,
        img_size: int = IMG_SIZE,
    ):
        super().__init__()
        self.enc = EncoderMLP(z=z, img_size=img_size)
        self.dec = DecoderMLP(z=z, img_size=img_size)
        self.n_classes = n_classes

        # Prototypes e_y in latent space
        self.protos = nn.Parameter(torch.randn(n_classes, z) * 0.1)

        self.register_buffer(
            "prior_logvar",
            torch.full((1, z), math.log(PRIOR_STD**2), dtype=torch.float32),
        )
        self.register_buffer(
            "sample_std",
            torch.full((1, z), float(sample_std), dtype=torch.float32),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        Returns:
            logits_inst: decoder logits from instance latent z_inst ~ q(z|x)
            mu, lv: encoder outputs for q(z|x)
            e_y: prototypes for the true labels y
        """
        mu, lv = self.enc(x)  # [B,z], [B,z]

        # Instance latent for reconstruction
        eps = torch.randn_like(mu)
        z_inst = mu + eps * torch.exp(0.5 * lv)
        logits_inst = self.dec(z_inst)

        # Prototypes for this batch's labels
        e_y = self.protos[y.long()]  # [B,z]

        return logits_inst, mu, lv, e_y

# ============================================================
# Loss: recon + KL(q||p(z|y)) + classification
# ============================================================

def kl_q_to_class_prior(mu: torch.Tensor,
                        lv: torch.Tensor,
                        e_y: torch.Tensor,
                        prior_logvar: torch.Tensor) -> torch.Tensor:
    """
    KL(q(z|x) || p(z|y)), where
      q(z|x) = N(mu, diag(exp(lv)))
      p(z|y) = N(e_y, diag(exp(prior_logvar)))

    Returns scalar (mean over batch).
    """
    # shapes:
    # mu, lv, e_y: [B, z]
    # prior_logvar: [1, z]
    var_q = lv.exp()               # [B,z]
    var_p = prior_logvar.exp()     # [1,z]

    # KL per sample (standard diagonal-Gaussian KL formula):
    # 0.5 * sum_j ( log(var_p_j / var_q_j)
    #              + (var_q_j + (mu_j - e_j)^2) / var_p_j
    #              - 1 )
    log_ratio = (prior_logvar - lv)             # [B,z]
    sq_diff = (mu - e_y).pow(2)                 # [B,z]
    frac = (var_q + sq_diff) / var_p            # [B,z]
    kl_per_sample = 0.5 * (log_ratio + frac - 1.0).sum(dim=1)  # [B]
    return kl_per_sample.mean()

def classification_loss_from_protos(mu: torch.Tensor,
                                    protos: torch.Tensor,
                                    y: torch.Tensor) -> torch.Tensor:
    """
    Likelihood-based classification:
        logits_c = -||mu - e_c||^2 / (2 * PRIOR_STD^2)
        p(y|x) = softmax(logits_c)

    Cross-entropy on these logits.
    """
    # mu: [B,z], protos: [C,z]
    mu_exp = mu.unsqueeze(1)        # [B,1,z]
    protos_exp = protos.unsqueeze(0) # [1,C,z]
    dist2 = (mu_exp - protos_exp).pow(2).sum(dim=2)  # [B,C]
    logits_cls = - dist2 / (2.0 * (PRIOR_STD**2))    # [B,C]
    return F.cross_entropy(logits_cls, y.long())

def proto_vae_loss(
    logits: torch.Tensor,
    x_ink01: torch.Tensor,
    mu: torch.Tensor,
    lv: torch.Tensor,
    e_y: torch.Tensor,
    prior_logvar: torch.Tensor,
    protos: torch.Tensor,
    y: torch.Tensor,
    beta_kl: float = BETA_KL,
    alpha_cls: float = ALPHA_CLS,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Total loss:
        recon + beta_kl * KL(q(z|x)||p(z|y)) + alpha_cls * classification_loss

    where classification loss is based on prototype likelihood.
    """
    # Reconstruction: Bernoulli pixels with logits
    recon = F.binary_cross_entropy_with_logits(logits, x_ink01, reduction="mean")

    # KL toward the class-specific prior N(e_y, PRIOR_STD^2 I)
    kl = kl_q_to_class_prior(mu, lv, e_y, prior_logvar)

    # Likelihood-based classification
    cls = classification_loss_from_protos(mu, protos, y)

    loss = recon + beta_kl * kl + alpha_cls * cls

    stats = {
        "recon": float(recon.item()),
        "kl": float(kl.item()),
        "cls": float(cls.item()),
    }
    return loss, stats

# ============================================================
# Sampling helpers
# ============================================================

def _get_default_std_from_model(model: nn.Module) -> float:
    return float(model.sample_std.view(-1)[0].item())

@torch.no_grad()
def sample_by_label(
    model: ProtoVAE,
    label_id: int,
    n_per: int = 8,
    device=DEVICE,
    sample_std: float | None = None,
):
    """
    Sample from the Gaussian around prototype e_{label_id}.
    """
    model.eval()
    e = model.protos[label_id].to(device).unsqueeze(0).expand(n_per, -1)
    if sample_std is None:
        sample_std = _get_default_std_from_model(model)
    z = e + sample_std * torch.randn_like(e)
    logits = model.dec(z)
    return torch.sigmoid(logits)  # [n_per,1,H,W], ink in [0,1]

# ============================================================
# Save generated images into ONE directory
# ============================================================

@torch.no_grad()
def save_singleproto_samples_individual(
    model: ProtoVAE,
    out_dir: str,
    n_per_proto: int = 10,
    device=DEVICE,
):
    """
    Save all single-prototype samples into ONE directory:
        out_dir/generated/
    Filenames: single_<classidx>_<label>_s<sample>.png
    """
    base_dir = os.path.join(out_dir, "generated")
    os.makedirs(base_dir, exist_ok=True)

    for c in range(model.n_classes):
        s = sample_by_label(
            model, c, n_per=n_per_proto,
            device=device, sample_std=GEN_SAMPLE_STD
        )
        # For visualization: 1 - s → white background, black ink
        imgs = 1.0 - s

        lbl_raw = LABELS[c]
        lbl_safe = safe_label_for_filename(lbl_raw)

        for k in range(n_per_proto):
            img = imgs[k:k+1]  # [1,1,H,W]
            fname = os.path.join(base_dir, f"single_{c:02d}_{lbl_safe}_s{k:03d}.png")
            save_image(img, fname, nrow=1, normalize=False)
    print(f"Saved single-proto samples in: {base_dir}")

# ============================================================
# Generative accuracy (nearest-neighbour in pixel space)
# ============================================================

@torch.no_grad()
def build_image_bank(
    ds: Dataset, device: torch.device = DEVICE, batch_size: int = 64
) -> Tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    all_x, all_y = [], []
    for x, y in loader:
        all_x.append(x.to(device))
        all_y.append(y.to(device))
    bank_imgs = torch.cat(all_x, dim=0)
    bank_labels = torch.cat(all_y, dim=0)
    return bank_imgs, bank_labels

@torch.no_grad()
def generative_accuracy_single_proto(
    model: ProtoVAE,
    ds: Dataset,
    n_samples_per_proto: int = 10,
    device: torch.device = DEVICE,
):
    """
    For each class c and its prototype:
        - sample n images from N(e_c, GEN_SAMPLE_STD^2 I),
        - decode,
        - find nearest neighbour in ds (pixelwise MSE),
        - predicted label = label of NN,
        - correct if predicted == c.

    Returns:
        acc_per_class: {label_name: acc}
        overall_acc: float
    """
    model.eval()
    bank_imgs, bank_labels = build_image_bank(ds, device=device)
    bank_flat = bank_imgs.view(bank_imgs.size(0), -1)

    C_total = model.n_classes
    acc_per_class: Dict[str, float] = {}
    correct_total = 0
    sample_total = 0

    for c in range(C_total):
        samples = sample_by_label(
            model, c, n_per=n_samples_per_proto,
            device=device, sample_std=GEN_SAMPLE_STD
        )
        samp_flat = samples.view(samples.size(0), -1)

        correct_c = 0
        for k in range(n_samples_per_proto):
            s = samp_flat[k : k + 1]              # [1,D]
            mse = (bank_flat - s).pow(2).mean(dim=1)  # [N]
            nn_idx = mse.argmin().item()
            pred_label = int(bank_labels[nn_idx].item())
            if pred_label == c:
                correct_c += 1

        acc = correct_c / float(n_samples_per_proto)
        acc_per_class[LABELS[c]] = acc
        correct_total += correct_c
        sample_total += n_samples_per_proto

    overall_acc = correct_total / float(sample_total)
    return acc_per_class, overall_acc

# ============================================================
# Training
# ============================================================

def train_proto_vae(
    model: ProtoVAE,
    ds: Dataset,
    epochs: int = EPOCHS,
    lr: float = LR,
    device: torch.device = DEVICE,
):
    loader = make_loader(ds, batch_size=BATCH_SIZE, shuffle=True)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        total_cls = 0.0
        n_samples = 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, mu, lv, e_y = model(x, y)

            loss, stats = proto_vae_loss(
                logits=logits,
                x_ink01=x,
                mu=mu,
                lv=lv,
                e_y=e_y,
                prior_logvar=model.prior_logvar,
                protos=model.protos,
                y=y,
                beta_kl=BETA_KL,
                alpha_cls=ALPHA_CLS,
            )

            opt.zero_grad()
            loss.backward()
            opt.step()

            bs = x.size(0)
            total_loss += loss.item() * bs
            total_recon += stats["recon"] * bs
            total_kl += stats["kl"] * bs
            total_cls += stats["cls"] * bs
            n_samples += bs

        avg_loss = total_loss / max(1, n_samples)
        avg_rec = total_recon / max(1, n_samples)
        avg_kl = total_kl / max(1, n_samples)
        avg_cls = total_cls / max(1, n_samples)
        print(
            f"[Epoch {ep:03d}] loss={avg_loss:.4f} "
            f"(rec={avg_rec:.4f}, kl={avg_kl:.4f}, cls={avg_cls:.4f})"
        )

# ============================================================
# Main
# ============================================================

def main():
    print(f"Using device: {DEVICE}")
    print(f"IMG_SIZE: {IMG_SIZE} x {IMG_SIZE}")

    print("Loading symbol-level datasets from paper splits...")
    train_ds = MathExprSymbolDataset(split="train")
    val_ds   = MathExprSymbolDataset(split="val")
    test_ds  = MathExprSymbolDataset(split="test")

    # ------------- ProtoVAE with 1 prototype per symbol -------------
    print("\n=== Training ProtoVAE (1 prototype per symbol) on TRAIN split ===")
    model = ProtoVAE(
        z=Z_DIM,
        n_classes=N_CLASSES,
        sample_std=TRAIN_SAMPLE_STD,
        img_size=IMG_SIZE,
    )
    train_proto_vae(model, train_ds, epochs=EPOCHS, lr=LR, device=DEVICE)

    print("\nSaving individual PNGs per prototype...")
    save_singleproto_samples_individual(model, OUT_DIR, n_per_proto=10, device=DEVICE)

    print("\nEvaluating generative accuracy on VAL split...")
    acc_per_class_val, overall_val = generative_accuracy_single_proto(
        model, val_ds, n_samples_per_proto=10, device=DEVICE
    )
    print(f"VAL overall generative accuracy: {overall_val:.4f}")
    for lbl, acc in acc_per_class_val.items():
        print(f"  {lbl:>4}: {acc:.4f}")

    print("\nEvaluating generative accuracy on TEST split...")
    acc_per_class_test, overall_test = generative_accuracy_single_proto(
        model, test_ds, n_samples_per_proto=10, device=DEVICE
    )
    print(f"TEST overall generative accuracy: {overall_test:.4f}")
    for lbl, acc in acc_per_class_test.items():
        print(f"  {lbl:>4}: {acc:.4f}")


if __name__ == "__main__":
    main()
