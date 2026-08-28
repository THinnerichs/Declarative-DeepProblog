#!/usr/bin/env python3
import os
import json
import argparse
from typing import Tuple

import numpy as np


# ---------- cost mapping 0..4 -> {0,1,4} ----------

def rebin_costs(vw: np.ndarray) -> np.ndarray:
    """
    Rebin vertex weights:
      0        -> 0
      1 or 2   -> 1
      3 or 4+  -> 4   (anything >= 3 is mapped to 4)
    Works elementwise for any shape.
    """
    x = vw.astype(np.int64)
    #  x <= 0  -> 0
    # 1 <= x <= 2 -> 1
    # x >= 3  -> 4
    y = np.where(x <= 0, 0, np.where(x <= 2, 1, 4))
    return y


# ---------- R/D/Diag shortest path DP (same as before, but self-contained) ----------

def rdd_shortest_cost(weights: np.ndarray, include_start: bool = True) -> int:
    """
    Dynamic programming shortest path with moves Right, Down, Diagonal (↘).
    Cost to enter a node = its vertex weight; include_start counts (0,0).
    Accepts a 2D (N,N) or 3D (N,N,1) array.
    """
    w = np.asarray(weights, dtype=np.int64)

    # If there is a singleton channel dimension, squeeze it.
    if w.ndim == 3 and w.shape[2] == 1:
        w = w[:, :, 0]

    if w.ndim != 2 or w.shape[0] != w.shape[1]:
        raise ValueError(f"rdd_shortest_cost: expected square 2D weights, got shape {w.shape}")

    N = w.shape[0]
    dp = np.zeros((N, N), dtype=np.int64)
    dp[0, 0] = int(w[0, 0]) if include_start else 0

    for j in range(1, N):
        dp[0, j] = dp[0, j - 1] + int(w[0, j])

    for i in range(1, N):
        dp[i, 0] = dp[i - 1, 0] + int(w[i, 0])

    for i in range(1, N):
        for j in range(1, N):
            dp[i, j] = min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1]) + int(w[i, j])

    return int(dp[N - 1, N - 1])


# ---------- helpers for handling stacked vs object arrays ----------

def rebin_vw_array(arr: np.ndarray) -> np.ndarray:
    """
    Apply rebin_costs to each map in a per-map array.
    Supports:
      - stacked array: shape (N, 3, 3) or (N, 3, 3, 1) etc.
      - object array: dtype=object, each entry is one per-map array.
    """
    if arr.dtype == object:
        out = [rebin_costs(x) for x in arr]
        return np.array(out, dtype=object)
    else:
        return rebin_costs(arr)


def recompute_sp_array(vw_arr: np.ndarray) -> np.ndarray:
    """
    Given per-map vertex weights (rebinned), compute one shortest-path cost per map.
    Returns a 1D int array of length N.
    """
    if vw_arr.dtype == object:
        costs = [rdd_shortest_cost(x) for x in vw_arr]
        return np.asarray(costs, dtype=np.int64)
    else:
        N = vw_arr.shape[0]
        costs = np.zeros(N, dtype=np.int64)
        for i in range(N):
            costs[i] = rdd_shortest_cost(vw_arr[i])
        return costs


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(
        description="Rebin vertex weights to {0,1,4} and recompute shortest paths for 3x3 Warcraft dataset."
    )
    ap.add_argument(
        "--in_dir",
        type=str,
        default="data/warcraft_shortest_path/3x3",
        help="Directory containing the existing 3x3 npy files",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="data/warcraft_shortest_path/3x3_rebinned",
        help="Output directory for the rebinned 3x3 dataset",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    splits = ["train", "val", "test"]
    for split in splits:
        maps_p_in = os.path.join(args.in_dir, f"{split}_maps.npy")
        vw_p_in   = os.path.join(args.in_dir, f"{split}_vertex_weights.npy")
        sp_p_in   = os.path.join(args.in_dir, f"{split}_shortest_paths.npy")

        maps_p_out = os.path.join(args.out_dir, f"{split}_maps.npy")
        vw_p_out   = os.path.join(args.out_dir, f"{split}_vertex_weights.npy")
        sp_p_out   = os.path.join(args.out_dir, f"{split}_shortest_paths.npy")

        # Copy maps unchanged (they're just images/tiles)
        if os.path.exists(maps_p_in):
            maps = np.load(maps_p_in, allow_pickle=True)
            np.save(maps_p_out, maps, allow_pickle=True)
            print(f"[ok] copied {split}_maps.npy: {maps.shape}")
        else:
            print(f"[skip] missing {maps_p_in}")

        # Rebin vertex weights
        if os.path.exists(vw_p_in):
            vw = np.load(vw_p_in, allow_pickle=True)
            vw_rebinned = rebin_vw_array(vw)
            np.save(vw_p_out, vw_rebinned, allow_pickle=True)
            print(f"[ok] {split}_vertex_weights.npy rebinned: {vw.shape} -> {vw_rebinned.shape}")
        else:
            print(f"[skip] missing {vw_p_in}")
            vw_rebinned = None

        # Recompute shortest paths from rebinned weights
        if vw_rebinned is not None:
            sp_re = recompute_sp_array(vw_rebinned)
            np.save(sp_p_out, sp_re, allow_pickle=True)
            print(f"[ok] {split}_shortest_paths.npy recomputed: {sp_re.shape}")
        else:
            print(f"[skip] cannot recompute shortest paths for split '{split}' (no vertex_weights)")

    # Copy & patch info.json if present
    info_in  = os.path.join(args.in_dir, "info.json")
    info_out = os.path.join(args.out_dir, "info.json")
    if os.path.exists(info_in):
        try:
            with open(info_in, "r") as f:
                info = json.load(f)
        except Exception:
            info = {}
        info["source_dir"] = os.path.relpath(args.in_dir, start=os.path.dirname(args.out_dir))
        info["grid_shape"] = [3, 3]
        info["cost_rebinning"] = {"0": 0, "1": 1, "2": 1, "3": 4, "4": 4}
        with open(info_out, "w") as f:
            json.dump(info, f, indent=2)
        print(f"[ok] wrote {info_out}")
    else:
        print(f"[skip] no info.json at {info_in}")


if __name__ == "__main__":
    main()
