# data/__init__.py
import os
import numpy as np
import torch
from typing import List
from deepproblog.dataset import Dataset, QueryDataset
from problog.logic import Term, Constant, Var

from deepproblog.query import Query

# -------------------- helpers --------------------

def _ensure_chw(x: np.ndarray) -> np.ndarray:
    """Return (C,H,W) for either (C,H,W) or (H,W,C)."""
    if x.ndim != 3:
        raise ValueError(f"_ensure_chw: expected 3D array, got {x.shape}")
    # HWC -> CHW
    if x.shape[-1] in (1, 3) and x.shape[0] not in (1, 3):
        return np.transpose(x, (2, 0, 1))
    return x

def _extract_tiles(m: np.ndarray, N: int) -> np.ndarray:
    """
    Convert a single map 'm' to tiled form (N,N,C,th,tw).
    Accepts:
      - (N,N,C,th,tw)
      - (N,N,th,tw,C)
      - (C,H,W) or (H,W,C) with H=W=N*tile
      - (N,N,th,tw)   (grayscale): will become (N,N,1,th,tw)
    """
    # Already tiled, channel-first
    if m.ndim == 5 and m.shape[0] == N and m.shape[1] == N:
        # (N,N,C,th,tw)
        if m.shape[2] in (1, 3) and m.shape[3] > 1 and m.shape[4] > 1:
            return m
        # (N,N,th,tw,C)
        if m.shape[-1] in (1, 3) and m.shape[2] > 1 and m.shape[3] > 1:
            return np.transpose(m, (0, 1, 4, 2, 3))
        # ambiguous 5D: last resort assume it's already (N,N,C,th,tw)
        return m

    # Grayscale tiled without channel dim: (N,N,th,tw)
    if m.ndim == 4 and m.shape[0] == N and m.shape[1] == N:
        th, tw = m.shape[2], m.shape[3]
        return m.reshape(N, N, 1, th, tw)

    # Full image: (C,H,W) or (H,W,C)
    if m.ndim == 3:
        chw = _ensure_chw(m)  # (C,H,W)
        C, H, W = chw.shape
        assert H % N == 0 and W % N == 0, f"Image not divisible by N={N}: {H}x{W}"
        th, tw = H // N, W // N
        tiles = np.zeros((N, N, C, th, tw), dtype=chw.dtype)
        for i in range(N):
            for j in range(N):
                tiles[i, j] = chw[:, i*th:(i+1)*th, j*tw:(j+1)*tw]
        return tiles

    raise ValueError(f"_extract_tiles: unrecognized map shape {m.shape} (ndim={m.ndim})")

def _extract_tile_from_map(m: np.ndarray, N: int, i: int, j: int) -> np.ndarray:
    """Return a single tile (C,th,tw) from map 'm'."""
    tiles = _extract_tiles(m, N)  # (N,N,C,th,tw)
    return tiles[i, j]

def _normalize_tile(tile: np.ndarray) -> np.ndarray:
    x = tile.astype(np.float32)
    if x.max() > 1.0:
        x = x / 255.0
    return x

# -------------------- R/D/Diag shortest path (for labels over N=3) --------------------

def rdd_shortest_cost(weights: np.ndarray, include_start: bool = True) -> int:
    """
    Dynamic programming shortest path with moves Right, Down, Diagonal (↘).
    Cost to enter a node = its vertex weight; include_start counts (0,0).
    """
    w = np.asarray(weights, dtype=np.int64)
    assert w.ndim == 2 and w.shape[0] == w.shape[1]
    N = w.shape[0]
    dp = np.zeros((N, N), dtype=np.int64)
    dp[0, 0] = int(w[0, 0]) if include_start else 0
    for j in range(1, N):
        dp[0, j] = dp[0, j-1] + int(w[0, j])
    for i in range(1, N):
        dp[i, 0] = dp[i-1, 0] + int(w[i, 0])
    for i in range(1, N):
        for j in range(1, N):
            dp[i, j] = min(dp[i-1, j], dp[i, j-1], dp[i-1, j-1]) + int(w[i, j])
    return int(dp[N-1, N-1])

# -------------------- tensor source --------------------

class MapTileSource:
    """
    Tensor source for DeepProbLog neural predicate input: tile(ID).
    Expects dataset under data_root/{N}x{N}/...
    ID = map_idx * (N*N) + i*N + j
    """
    def __init__(self, split: str, N: int = 12, data_root: str = "data/warcraft_shortest_path"):
        self.N = N
        self.split = split
        self.base = os.path.join(data_root, f"{N}x{N}")
        self.maps = np.load(os.path.join(self.base, f"{split}_maps.npy"), allow_pickle=True)
        self.vw   = np.load(os.path.join(self.base, f"{split}_vertex_weights.npy"), allow_pickle=True)

        # Infer (C,th,tw) from the first tile of the first map
        t0 = _extract_tile_from_map(self.maps[0], self.N, 0, 0)  # (C,th,tw)
        self.CHW = tuple(t0.shape)

    @property
    def num_maps(self) -> int:
        return len(self.maps)

    def __call__(self, global_id: int) -> torch.Tensor:
        N = self.N
        m = global_id // (N * N)
        r = global_id %  (N * N)
        i = r // N
        j = r %  N
        tile = _extract_tile_from_map(self.maps[m], N, i, j)
        tile = _normalize_tile(tile)
        return torch.from_numpy(tile)  # (C,th,tw) float32

# -------------------- datasets --------------------

class WarcraftTiles(Dataset):
    """
    Tile-level supervision:
      nn(cost_net, [Tile], Cost) :: tile_cost(Tile, Cost).
    Builds queries: tile_cost(tile(ID), Label) with Label from vertex_weights.
    """
    def __init__(self, split: str, source_name: str, tile_source: MapTileSource, N: int = 12):
        super().__init__()
        self.split = split
        self.N = N
        self.source = tile_source
        self._queries: List[Term] = []

        for m in range(self.source.num_maps):
            vw = np.array(self.source.vw[m])
            for i in range(N):
                for j in range(N):
                    gid = m * (N * N) + i * N + j
                    y = Constant(int(vw[i, j]))
                    self._queries.append(Term("tile_cost", Term("tile", Constant(gid)), y))

    def __len__(self) -> int:
        return len(self._queries)

    def to_query(self, i: int) -> Term:
        return self._queries[i]

class WarcraftSP_RDD_Maps(Dataset):
    """
    Map-level dataset for end-to-end training (N=3):
      sp_cost(map(M), Y)
    Gold Y is the R/D/Diag shortest-path sum computed from vertex_weights.
    """
    def __init__(self, split: str, source_name: str, tile_source: MapTileSource, N: int = 3, labeled: bool = True):
        super().__init__()
        self.split = split
        self.N = N
        self.source = tile_source
        self.labeled = labeled

        self._labels = [rdd_shortest_cost(np.array(self.source.vw[m])) for m in range(self.source.num_maps)]
        self._queries: List[Term] = []
        for m, y in enumerate(self._labels):
            y_term = Constant(int(y)) if labeled else Var("Y")
            self._queries.append(Term("sp_cost", Term("map", Constant(m)), y_term))

    def __len__(self) -> int:
        return len(self._queries)

    def to_query(self, i: int) -> Query:
        return Query(self._queries[i])

    def gold_labels(self) -> np.ndarray:
        return np.asarray(self._labels, dtype=np.int64)

    def peek(self, n: int = 5):
        print(f"\n[{self.split} | N={self.N}] first {n} R/D/Diag shortest-path queries:")
        for k in range(min(n, len(self._queries))):
            print(f"  {k:3d}: sp_cost(map({k}), {int(self._labels[k]) if self.labeled else 'Y'})")
        print()
