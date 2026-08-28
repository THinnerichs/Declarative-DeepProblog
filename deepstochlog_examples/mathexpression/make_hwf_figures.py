import os
import random
from typing import List, Tuple, Dict

import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from PIL import Image

from mathexpression_data import (
    AbstractMathExprDataset,
    all_symbols_list,
    digit_idx_list,
    operator_idx_list,
)

# ------------ config ------------

SEED = 0
random.seed(SEED)

IMG_SIZE = 45  # should match dataset
OUT_FIG_DIR = "figures_hwf"
os.makedirs(OUT_FIG_DIR, exist_ok=True)

# Path to crisp prototype samples
CRISP_DIR = (
    "/home/thinnerichs/Documents/Work/Delft/Projects/NeSy/DeclDeepProblog/"
    "generative-deepproblog/deepstochlog_examples/mathexpression/"
    "out_vaehwf_splits/generated/generated_crisp"
)

LABELS: List[str] = all_symbols_list  # label_id -> string (e.g. "1", "+", ...)

# ------------ dataset wrapper: symbol-level ------------

class MathExprSymbolDataset(AbstractMathExprDataset, Dataset):
    """
    Symbol-level dataset: each symbol image in each expression is one sample.
    Returns (image, label_id) where image is [1,H,W] in [0,1], 1=ink.
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
        x_norm = self.tensors[img_id]   # [1,H,W], in [-0.5,0.5]

        # Map to [0,1] with 1=ink, 0=background
        x = 0.5 - x_norm
        x = x.clamp(0.0, 1.0)

        return x, int(label)

# ------------ mapping helpers ------------

def token_to_label(token: str) -> str:
    """
    Map prototype filename token to dataset label string.

    prototype_3_000.png    -> token="3"    -> label "3"
    prototype_plus_000.png -> token="plus" -> label "+"
    prototype_minus_000.png-> token="minus"-> label "-"
    prototype_times_000.png-> token="times"-> label "*"
    prototype_div_000.png  -> token="div"  -> label "/"
    """
    if token in [str(i) for i in range(10)]:
        return token
    mapping = {
        "plus": "+",
        "minus": "-",
        "times": "*",
        "div": "/",
    }
    return mapping.get(token, token)

def label_to_token(lbl: str) -> str:
    """
    Map dataset label string to prototype token.

    "3" -> "3"
    "+" -> "plus"
    "-" -> "minus"
    "*" -> "times"
    "/" -> "div"
    """
    if lbl in [str(i) for i in range(10)]:
        return lbl
    mapping = {
        "+": "plus",
        "-": "minus",
        "*": "times",
        "/": "div",
    }
    return mapping.get(lbl, lbl)

# ------------ helper: load crisp images per label ------------

def index_crisp_images(crisp_dir: str) -> Dict[str, List[str]]:
    """
    Build a mapping: dataset_label_string -> list of file paths.

    Filenames look like:
        prototype_0_000.png
        prototype_3_007.png
        prototype_plus_002.png
        prototype_minus_005.png
        prototype_times_004.png
        prototype_div_009.png

    We parse the token between 'prototype_' and the next '_' and
    map that to the dataset label string.
    """
    files = [
        f for f in os.listdir(crisp_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
        and f.startswith("prototype_")
    ]
    mapping: Dict[str, List[str]] = {}

    for f in files:
        # pattern: prototype_<token>_<idx>.png
        core = f[len("prototype_"):]
        parts = core.split("_")
        if len(parts) < 2:
            continue
        token = parts[0]
        label_str = token_to_label(token)
        full = os.path.join(crisp_dir, f)
        mapping.setdefault(label_str, []).append(full)

    return mapping

# ------------ figure 1: original images grid ------------

def make_figure1_originals(ds: MathExprSymbolDataset, save_path: str):
    """
    4x8 grid (32 images), without labels / headers:
      - first 3 rows: digits
      - last row: operators

    Returns the list of (img_tensor, label_id) in raster-scan order.
    """
    # collect indices for digits and operators
    digit_indices = []
    op_indices = []
    for idx in range(len(ds)):
        _, y = ds[idx]
        if y in digit_idx_list:
            digit_indices.append(idx)
        elif y in operator_idx_list:
            op_indices.append(idx)

    # random sub-selection
    n_digits = 3 * 8   # 3 rows * 8 columns
    n_ops    = 1 * 8   # 1 row  * 8 columns

    chosen_digit_idxs = random.sample(digit_indices, n_digits)
    chosen_op_idxs    = random.sample(op_indices,   n_ops)

    # load the actual tensors, in order: digits first, then operators
    selection: List[Tuple[torch.Tensor, int]] = []
    for idx in chosen_digit_idxs:
        x, y = ds[idx]
        selection.append((x, y))
    for idx in chosen_op_idxs:
        x, y = ds[idx]
        selection.append((x, y))

    # plot
    fig, axes = plt.subplots(4, 8, figsize=(8, 4))
    # no suptitle, no per-ax titles

    for i in range(4):
        for j in range(8):
            ax = axes[i, j]
            k = i * 8 + j
            x, y = selection[k]
            img = x.squeeze(0).cpu().numpy()
            # 1 - img: white background, black ink
            ax.imshow(1.0 - img, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
            ax.axis("off")

    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved figure 1 (originals) to {save_path}")
    return selection

# ------------ figure 2: prototype samples for those labels ------------

def make_figure2_prototype_samples(
    selection: List[Tuple[torch.Tensor, int]],
    crisp_map: Dict[str, List[str]],
    save_path: str,
):
    """
    4x8 grid, same layout as figure 1, but each cell shows a random
    crisp prototype sample for the corresponding label.
    No labels or headers.
    """
    fig, axes = plt.subplots(4, 8, figsize=(8, 4))

    for k, (x, y) in enumerate(selection):
        i = k // 8
        j = k % 8
        ax = axes[i, j]

        lbl = LABELS[y]   # dataset label string (e.g. "3", "+")
        paths = crisp_map.get(lbl, [])
        if not paths:
            ax.axis("off")
            continue
        img_path = random.choice(paths)
        img = Image.open(img_path).convert("L")
        ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.axis("off")

    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved figure 2 (prototype samples) to {save_path}")

def make_figure3_prototype_grid(
    crisp_map: Dict[str, List[str]],
    save_path: str,
):
    """
    4x9 grid:

      columns: digits 1–5 and the 4 operators (+, -, *, /)
      rows:    different crispified samples for that prototype

    No labels / headers.
    """
    symbols = ["1", "2", "3", "4", "5", "+", "-", "*", "/"]
    n_rows = 4
    n_cols = len(symbols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))

    for j, sym in enumerate(symbols):
        paths = crisp_map.get(sym, [])
        for i in range(n_rows):
            ax = axes[i, j]
            if not paths:
                ax.axis("off")
                continue
            img_path = random.choice(paths)
            img = Image.open(img_path).convert("L")
            ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.axis("off")

    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved figure 3 (prototype grid) to {save_path}")

def main():
    # 1) Load validation set as symbol-level dataset
    val_ds = MathExprSymbolDataset(split="val")

    # 2) Figure 1: originals
    fig1_path = os.path.join(OUT_FIG_DIR, "figure1_originals_4x8.png")
    selection = make_figure1_originals(val_ds, fig1_path)

    # 3) Index crisp prototype images
    crisp_map = index_crisp_images(CRISP_DIR)
    print("Crisp images per label (found in directory):")
    for lbl, paths in crisp_map.items():
        print(f"  {repr(lbl)}: {len(paths)} files")

    # 4) Figure 2: prototype samples matching Figure 1 labels
    fig2_path = os.path.join(OUT_FIG_DIR, "figure2_prototypes_for_originals_4x8.png")
    make_figure2_prototype_samples(selection, crisp_map, fig2_path)

    # 5) Figure 3: prototype grid 4x9
    fig3_path = os.path.join(OUT_FIG_DIR, "figure3_prototype_grid_4x9.png")
    make_figure3_prototype_grid(crisp_map, fig3_path)


if __name__ == "__main__":
    main()
