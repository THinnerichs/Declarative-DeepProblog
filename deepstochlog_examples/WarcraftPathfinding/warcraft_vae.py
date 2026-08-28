# warcraft_vae.py
#
# Prototype-based VAE-ish model for Warcraft tiles:
#   - one learnable latent prototype per tile cost (0..4)
#   - shared decoder: prototype -> canonical tile image
#   - trained by reconstruction loss on real tiles
#
# Map generation:
#   - enumerate 3x3 grids over tile costs {0..4}
#   - keep layouts whose R/D/Diag shortest-path cost == target_cost
#   - select a DIVERSE subset w.r.t. the actual path (unique paths)
#   - decode each tile from the corresponding prototype
#   - stitch tiles into a full map and save:
#       * plain map PNG
#       * map with the shortest path drawn on top (second PNG)

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path
from typing import List, Tuple, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from data import _extract_tiles, _normalize_tile, rdd_shortest_cost


# -------------------------------------------------------
# Dataset: per-tile reconstruction with cost labels 0..4
# -------------------------------------------------------

class WarcraftTileCostDataset(Dataset):
    """
    Each item is a single tile + its *binned* cost label (0..4).
    For N=3: each map yields 9 (tile, label) pairs.

    Vertex weights in the .npy may be larger than 4; those are clamped to 4.
    """

    def __init__(
        self,
        split: str = "train",
        data_root: str = "data/warcraft_shortest_path",
        N: int = 3,
    ):
        super().__init__()
        self.N = N
        base = Path(data_root) / f"{N}x{N}"

        self.maps = np.load(base / f"{split}_maps.npy", allow_pickle=True)
        self.vw   = np.load(base / f"{split}_vertex_weights.npy", allow_pickle=True)

        self.maps = np.asarray(self.maps)
        self.vw   = np.asarray(self.vw)

        assert len(self.maps) == len(self.vw), "maps and vertex_weights size mismatch"

        # We *decide* that tile costs are in {0,1,2,3,4}, with any higher values clamped to 4
        self.cost_values: List[int] = [0, 1, 2, 3, 4]

    @property
    def num_classes(self) -> int:
        return len(self.cost_values)

    def __len__(self) -> int:
        return len(self.maps) * self.N * self.N

    def __getitem__(self, idx: int):
        N = self.N
        m_idx = idx // (N * N)
        r     = idx %  (N * N)
        i     = r // N
        j     = r %  N

        # Map image + vertex weights for that map
        m = self.maps[m_idx]
        w = self.vw[m_idx]

        # Extract tile image
        tiles = _extract_tiles(m, N)       # (N,N,C,th,tw) or (N,N,th,tw,C)
        tile = tiles[i, j]
        tile = _normalize_tile(tile).astype(np.float32)  # (C,th,tw) in [0,1]

        # Raw vertex cost -> clamp to [0,4]
        raw_cost = int(w[i, j])
        cost = max(0, min(raw_cost, 4))

        x = torch.from_numpy(tile)                 # (C,th,tw)
        y = torch.tensor(cost, dtype=torch.long)   # scalar in {0..4}
        return x, y


# -------------------------------------------------------
# Tiling helper: stitch tiles into a full map
# -------------------------------------------------------

def stitch_torch_tiles(tiles: torch.Tensor, N: int) -> torch.Tensor:
    """
    tiles: (N*N, C, th, tw)
    return: (C, N*th, N*tw)
    """
    T, C, th, tw = tiles.shape
    assert T == N * N, f"Expected {N*N} tiles, got {T}"
    tiles = tiles.view(N, N, C, th, tw)       # (N,N,C,th,tw)
    tiles = tiles.permute(2, 0, 3, 1, 4)      # (C,N,th,N,tw)
    tiles = tiles.reshape(C, N * th, N * tw)  # (C,H,W)
    return tiles


# -------------------------------------------------------
# Prototype-decoder model
# -------------------------------------------------------

class TileDecoder(nn.Module):
    """
    Simple decoder: latent_dim -> (C,8,8) tile in [0,1].

    This assumes tiles are around 8x8; if they are slightly different size,
    we up/downsample at the end.
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
            nn.Conv2d(16, out_ch, 3, padding=1),                 # (C,8,8)
            nn.Sigmoid(),                                       # [0,1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)                              # (B,16*4*4)
        h = h.view(z.size(0), 16, 4, 4)             # (B,16,4,4)
        x_hat = self.deconv(h)                      # (B,C,8,8)

        if x_hat.shape[-2:] != (self.tile_h, self.tile_w):
            x_hat = F.interpolate(
                x_hat,
                size=(self.tile_h, self.tile_w),
                mode="bilinear",
                align_corners=False,
            )
        return x_hat


class CostPrototypeDecoder(nn.Module):
    """
    One learnable latent prototype per cost class (0..4), shared decoder.

    For a tile of cost y, we decode the prototype z_y and try to reconstruct x.
    Training loss: MSE(x, decode(proto[y])).

    Purely generative; no encoder / classifier here.
    """

    def __init__(
        self,
        n_classes: int,
        latent_dim: int,
        out_ch: int,
        tile_h: int,
        tile_w: int,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.latent_dim = latent_dim

        # One prototype latent per class, learnable
        self.prototype = nn.Parameter(
            0.1 * torch.randn(n_classes, latent_dim)
        )  # (K,D)

        # Shared decoder
        self.decoder = TileDecoder(latent_dim, out_ch, tile_h, tile_w)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        y: (B,) long, cost labels in [0, n_classes-1]
        returns:
          recon: (B,C,th,tw)
        """
        z = self.prototype[y]          # (B,D)
        recon = self.decoder(z)        # (B,C,th,tw)
        return recon

    @torch.no_grad()
    def decode_class(self, cls_idx: int, device: torch.device) -> torch.Tensor:
        """
        Decode the canonical tile for a given cost class.
        returns: (C,th,tw)
        """
        idx = torch.tensor([cls_idx], dtype=torch.long, device=device)
        z = self.prototype[idx]             # (1,D)
        tile = self.decoder(z)[0]           # (C,th,tw)
        return tile.clamp(0.0, 1.0)


# -------------------------------------------------------
# Training
# -------------------------------------------------------

def train_cost_prototypes(
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    N: int = 3,
    data_root: str = "data/warcraft_shortest_path",
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[CostPrototypeDecoder, WarcraftTileCostDataset]:
    """
    Train the prototype-decoder model on per-tile reconstruction.
    """
    device = torch.device(device)

    train_ds = WarcraftTileCostDataset(split="train", N=N, data_root=data_root)
    val_ds   = WarcraftTileCostDataset(split="val",   N=N, data_root=data_root)

    # Peek at shapes
    x0, y0 = train_ds[0]
    C, th, tw = x0.shape
    K = train_ds.num_classes
    print(f"Tiles: N={N}, C={C}, th={th}, tw={tw}, num_classes={K}, example cost={y0.item()}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    model = CostPrototypeDecoder(
        n_classes=K,
        latent_dim=16,
        out_ch=C,
        tile_h=th,
        tile_w=tw,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for x, y in train_loader:
            x = x.to(device)    # (B,C,th,tw)
            y = y.to(device)    # (B,)

            recon = model(y)    # (B,C,th,tw)
            loss = F.mse_loss(recon, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

        train_loss = total_loss / total_samples

        # quick validation
        model.eval()
        val_loss = 0.0
        val_samples = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                recon = model(y)
                loss = F.mse_loss(recon, x)
                val_loss += loss.item() * x.size(0)
                val_samples += x.size(0)
        val_loss = val_loss / max(1, val_samples)

        print(
            f"Epoch {epoch:02d}: "
            f"train_recon_loss={train_loss:.6f}, "
            f"val_recon_loss={val_loss:.6f}"
        )

    return model, train_ds


# -------------------------------------------------------
# Shortest-path with actual path reconstruction
# -------------------------------------------------------

def shortest_path_rdd_with_path(
    weights: np.ndarray,
    include_start: bool = True,
) -> Tuple[int, List[Tuple[int, int]]]:
    """
    Dynamic programming R/D/Diag shortest path with path reconstruction.

    Returns:
      cost: int
      path: list of (i,j) from (0,0) to (N-1,N-1)
    """
    w = np.asarray(weights, dtype=np.int64)
    assert w.ndim == 2 and w.shape[0] == w.shape[1]
    N = w.shape[0]

    dp = np.full((N, N), np.iinfo(np.int64).max, dtype=np.int64)
    parent = np.full((N, N, 2), -1, dtype=np.int64)

    dp[0, 0] = int(w[0, 0]) if include_start else 0
    parent[0, 0] = (-1, -1)

    # First row
    for j in range(1, N):
        dp[0, j] = dp[0, j-1] + int(w[0, j])
        parent[0, j] = (0, j-1)

    # First column
    for i in range(1, N):
        dp[i, 0] = dp[i-1, 0] + int(w[i, 0])
        parent[i, 0] = (i-1, 0)

    # Inner cells
    for i in range(1, N):
        for j in range(1, N):
            candidates = [
                (dp[i, j-1], (i, j-1)),   # from left
                (dp[i-1, j], (i-1, j)),   # from above
                (dp[i-1, j-1], (i-1, j-1)) # from diag
            ]
            best_cost, best_parent = min(candidates, key=lambda x: x[0])
            dp[i, j] = best_cost + int(w[i, j])
            parent[i, j] = best_parent

    # Reconstruct path
    path: List[Tuple[int, int]] = []
    ci, cj = N - 1, N - 1
    while ci >= 0 and cj >= 0:
        path.append((ci, cj))
        pi, pj = parent[ci, cj]
        if pi < 0 or pj < 0:
            break
        ci, cj = pi, pj

    path.reverse()
    return int(dp[N-1, N-1]), path


# -------------------------------------------------------
# Diverse layout enumeration & decoding
# -------------------------------------------------------

def diverse_layouts_for_cost(
    target_cost: int,
    N: int = 3,
    cost_values: Sequence[int] | None = None,
    num_desired: int = 10,
    max_search: int | None = None,
) -> List[Tuple[np.ndarray, List[Tuple[int, int]]]]:
    """
    Enumerate 3x3 grids over cost_values whose R/D/Diag shortest-path cost equals target_cost,
    but keep a DIVERSE set: at most one layout per unique path.

    Returns:
      list of (layout, path), where:
        - layout: (N,N) np.ndarray of costs
        - path:   list of (i,j) from (0,0) to (N-1,N-1)
    """
    if cost_values is None:
        cost_values = [0, 1, 2, 3, 4]
    cost_values = list(cost_values)

    if max_search is None:
        max_search = len(cost_values) ** (N * N)

    results: List[Tuple[np.ndarray, List[Tuple[int, int]]]] = []
    seen_paths: set[Tuple[Tuple[int, int], ...]] = set()

    for idx, flat in enumerate(itertools.product(cost_values, repeat=N * N)):
        if idx >= max_search:
            break

        grid = np.array(flat, dtype=np.int64).reshape(N, N)
        cost, path = shortest_path_rdd_with_path(grid, include_start=True)
        if cost != target_cost:
            continue

        path_tuple = tuple(path)
        if path_tuple in seen_paths:
            continue

        seen_paths.add(path_tuple)
        results.append((grid, path))

        if len(results) >= num_desired:
            break

    return results


@torch.no_grad()
def decode_layout_to_map(
    model: CostPrototypeDecoder,
    layout: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """
    layout: (N,N) array of tile costs in {0..4}
    returns: map tensor (C,H,W) in [0,1]
    """
    N = layout.shape[0]
    device = torch.device(device)

    tiles: List[torch.Tensor] = []
    for i in range(N):
        for j in range(N):
            cost = int(layout[i, j])
            tile = model.decode_class(cost, device=device)  # (C,th,tw)
            tiles.append(tile)

    tiles_t = torch.stack(tiles, dim=0)            # (N*N,C,th,tw)
    recon_map = stitch_torch_tiles(tiles_t, N)     # (C,H,W)
    return recon_map.clamp(0.0, 1.0)


def save_map_tensor(
    m: torch.Tensor,
    path: Path,
):
    """
    m: (C,H,W) in [0,1]
    """
    m_np = (m.clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
    if m_np.shape[0] == 1:
        # grayscale
        img = Image.fromarray(m_np[0], mode="L")
    else:
        img = Image.fromarray(np.transpose(m_np, (1, 2, 0)), mode="RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def draw_path_on_map(
    m: torch.Tensor,
    path: List[Tuple[int, int]],
    N: int,
    tile_h: int,
    tile_w: int,
    alpha: float = 0.6,
) -> torch.Tensor:
    """
    Overlay the shortest path onto the map by tinting the corresponding tiles red.

    m: (C,H,W) in [0,1]
    path: list of (i,j) tile indices
    """
    overlay = m.clone()
    C, H, W = overlay.shape
    assert H == N * tile_h and W == N * tile_w, "Inconsistent tile sizes"

    for (i, j) in path:
        y0 = i * tile_h
        y1 = (i + 1) * tile_h
        x0 = j * tile_w
        x1 = (j + 1) * tile_w

        if C >= 3:
            # Red overlay: emphasize red, suppress green/blue
            base = overlay[:, y0:y1, x0:x1]
            red     = base[0]
            green   = base[1]
            blue    = base[2]

            red   = (1 - alpha) * red + alpha * 1.0
            green = (1 - alpha) * green
            blue  = (1 - alpha) * blue

            base[0] = red.clamp(0.0, 1.0)
            base[1] = green.clamp(0.0, 1.0)
            base[2] = blue.clamp(0.0, 1.0)
            overlay[:, y0:y1, x0:x1] = base
        else:
            # Grayscale: darken the path tiles
            base = overlay[0, y0:y1, x0:x1]
            base = (1 - alpha) * base
            overlay[0, y0:y1, x0:x1] = base.clamp(0.0, 1.0)

    return overlay


def generate_maps_for_cost(
    model: CostPrototypeDecoder,
    target_cost: int,
    out_dir: str = "output/gen_maps",
    N: int = 3,
    num_maps: int = 5,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Given a trained prototype model, generate image maps whose R/D/Diag shortest path
    has the desired total cost.

    For each generated map:
      - save plain map
      - save map with the shortest path drawn in red
    """
    device = torch.device(device)
    cost_values = [0, 1, 2, 3, 4]

    print(f"Searching for diverse layouts with target shortest-path cost {target_cost}...")
    layouts_with_paths = diverse_layouts_for_cost(
        target_cost=target_cost,
        N=N,
        cost_values=cost_values,
        num_desired=num_maps,
        max_search=None,  # search entire space if needed
    )

    if not layouts_with_paths:
        print(f"No {N}x{N} layouts found for shortest cost = {target_cost}")
        return

    print(f"Found {len(layouts_with_paths)} diverse layouts for cost {target_cost}.")
    out_dir = Path(out_dir)

    # We know decoder tile size from the model
    dummy_tile = model.decode_class(0, device=device)  # (C,th,tw)
    _, th, tw = dummy_tile.shape

    for k, (layout, path) in enumerate(layouts_with_paths[:num_maps]):
        m = decode_layout_to_map(model, layout, device=device)           # (C,H,W)
        fname_base = out_dir / f"cost_{target_cost:03d}_map_{k:03d}"

        # 1) Plain map
        save_map_tensor(m, fname_base.with_suffix(".png"))

        # 2) Map with path overlay
        m_path = draw_path_on_map(m, path, N=N, tile_h=th, tile_w=tw, alpha=0.6)
        save_map_tensor(m_path, fname_base.with_name(fname_base.name + "_path.png"))

        print(f"Saved {fname_base.with_suffix('.png')} and {fname_base.name + '_path.png'} with layout:\n{layout}")
        print(f"Path: {path}")


# -------------------------------------------------------
# CLI
# -------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Prototype-based VAE for Warcraft tiles + cost-based map generation")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--N", type=int, default=3)
    p.add_argument("--data_root", type=str, default="data/warcraft_shortest_path")
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--gen_cost",
        type=int,
        default=None,
        help="If set, generate maps whose shortest-path cost equals this value after training.",
    )
    p.add_argument("--gen_num", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model, train_ds = train_cost_prototypes(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        N=args.N,
        data_root=args.data_root,
        device=device,
    )

    if args.gen_cost is not None:
        generate_maps_for_cost(
            model=model,
            target_cost=args.gen_cost,
            num_maps=args.gen_num,
            N=args.N,
            device=device,
        )


if __name__ == "__main__":
    main()
