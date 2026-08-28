#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
from typing import Tuple

# --------- Helpers for tile handling ---------
def _ensure_chw(x: np.ndarray) -> np.ndarray:
    """Return array as (C,H,W) if it is either (H,W,C) or (C,H,W)."""
    if x.ndim != 3:
        raise ValueError(f"Expected 3D array, got {x.shape}")
    # If last dim is channels, assume HWC -> CHW
    if x.shape[-1] in (1, 3) and x.shape[0] not in (1, 3):
        return np.transpose(x, (2, 0, 1))
    return x

def _extract_tiles(m: np.ndarray, grid_hw: Tuple[int,int]=(12,12), tile_hw: Tuple[int,int]=(8,8)) -> np.ndarray:
    """
    Return tiles shaped (Hgrid,Wgrid,C,th,tw).
    Accepts:
      - already tiled: (Hgrid,Wgrid,C,th,tw) or (Hgrid,Wgrid,th,tw,C)
      - full image (C,H,W) or (H,W,C) where H = Hgrid*th, W = Wgrid*tw.
    """
    Hg, Wg = grid_hw
    th, tw = tile_hw

    if m.ndim == 5:
        # (Hg,Wg,C,th,tw) or (Hg,Wg,th,tw,C)
        if m.shape[:2] != (Hg, Wg):
            raise ValueError(f"Unexpected tiled grid shape {m.shape[:2]}, expected {(Hg,Wg)}")
        if m.shape[2:] == (th, tw, 3) or m.shape[2:] == (th, tw, 1):
            # (Hg,Wg,th,tw,C) -> (Hg,Wg,C,th,tw)
            return np.transpose(m, (0,1,4,2,3))
        elif m.shape[2:] == (3, th, tw) or m.shape[2:] == (1, th, tw):
            # (Hg,Wg,C,th,tw)
            return m
        else:
            # assume already (Hg,Wg,C,th,tw)
            return m

    if m.ndim == 3:
        chw = _ensure_chw(m)  # (C,H,W)
        C,H,W = chw.shape
        if H % Hg != 0 or W % Wg != 0:
            raise ValueError(f"Image size {H}x{W} not divisible by grid {Hg}x{Wg}")
        th_guess, tw_guess = H // Hg, W // Wg
        if (th, tw) != (th_guess, tw_guess):
            # trust what's implied by the image
            th, tw = th_guess, tw_guess
        tiles = np.zeros((Hg, Wg, C, th, tw), dtype=chw.dtype)
        for i in range(Hg):
            for j in range(Wg):
                tiles[i,j] = chw[:, i*th:(i+1)*th, j*tw:(j+1)*tw]
        return tiles

    raise ValueError(f"Unrecognized map array shape: {m.shape}")

def _tiles_to_full_image(tiles: np.ndarray) -> np.ndarray:
    """(Hg,Wg,C,th,tw) -> full image (H,W,C) in the same dtype."""
    assert tiles.ndim == 5
    Hg, Wg, C, th, tw = tiles.shape
    H, W = Hg*th, Wg*tw
    full = np.zeros((C, H, W), dtype=tiles.dtype)
    for i in range(Hg):
        for j in range(Wg):
            full[:, i*th:(i+1)*th, j*tw:(j+1)*tw] = tiles[i,j]
    return np.transpose(full, (1,2,0))  # HWC

def _reformat_like(original: np.ndarray, cropped_tiles: np.ndarray) -> np.ndarray:
    """
    Given the original per-map array and its cropped tiles (R,C,Ch,th,tw),
    return a per-map array with the same 'style' as the original.
    """
    if original.ndim == 3:
        # Original was a full image. Rebuild a full image (H,W,C).
        img_hwc = _tiles_to_full_image(cropped_tiles)  # (H,W,C)
        # If original looked like CHW, transpose back.
        if original.shape[0] in (1,3) and original.shape[-1] not in (1,3):
            return np.transpose(img_hwc, (2,0,1))  # CHW
        return img_hwc  # HWC

    if original.ndim == 5:
        # Original was tiled. Match channel-last vs channel-first.
        if original.shape[2:4] == cropped_tiles.shape[3:5] and original.shape[-1] in (1,3):
            # original (Hg,Wg,th,tw,C) -> keep that layout
            return np.transpose(cropped_tiles, (0,1,3,4,2))
        else:
            # assume (Hg,Wg,C,th,tw)
            return cropped_tiles

    # Fallback: return cropped tiles
    return cropped_tiles

# --------- Cropping utilities ---------
def compute_origin(origin: str, grid_hw: Tuple[int,int], crop_hw: Tuple[int,int]) -> Tuple[int,int]:
    Hg, Wg = grid_hw
    Rh, Rw = crop_hw
    if isinstance(origin, str):
        o = origin.lower()
        if o in ("tl", "top-left", "topleft"):
            return 0, 0
        if o in ("center", "centre", "c"):
            return (Hg - Rh)//2, (Wg - Rw)//2
        if o in ("br", "bottom-right", "bottomright"):
            return Hg - Rh, Wg - Rw
    raise ValueError(f"Unrecognized origin '{origin}'. Use 'top-left', 'center', or 'bottom-right'.")

def crop_tiles(tiles: np.ndarray, i0: int, j0: int, Rh: int, Rw: int) -> np.ndarray:
    """tiles (Hg,Wg,C,th,tw) -> (Rh,Rw,C,th,tw)"""
    return tiles[i0:i0+Rh, j0:j0+Rw, :, :, :]

def crop_vertex_weights(arr):
    """
    Try to crop vertex weights for a single map.
    Expected common shapes:
      - (12,12)
      - (12,12,1) or (12,12,C)
    """
    if arr.ndim == 2 and arr.shape == (12,12):
        return arr[:3,:3]
    if arr.ndim == 3 and arr.shape[:2] == (12,12):
        return arr[:3,:3,...]
    # Unknown layout; return as-is
    return arr

def crop_shortest_paths(sp):
    """
    Crop a single-map shortest_paths array to the 3x3 subgrid.

    Supported per-map shapes:
      - (12,12)                 -> [:3,:3]
      - (12,12,...)             -> [:3,:3,...]   (e.g., channel dims)
      - (12,12,12,12)           -> [:3,:3,:3,:3]
      - (144,144)               -> reindex to the 9 nodes of the 3x3 top-left subgrid
      - dtype=object containers of the above
    """
    # 4D tensor of all-pairs paths on the grid
    if sp.ndim == 4 and sp.shape[:4] == (12,12,12,12):
        return sp[:3, :3, :3, :3]

    # 2D adjacency (flattened 12x12 -> 144 nodes)
    if sp.ndim == 2 and sp.shape == (144,144):
        idx = [i*12 + j for i in range(3) for j in range(3)]  # top-left 3x3 indices
        return sp[np.ix_(idx, idx)]

    # 12x12 grid (mask or cost-like), possibly with extra trailing dims
    if sp.ndim >= 2 and sp.shape[0] == 12 and sp.shape[1] == 12:
        slicer = (slice(0,3), slice(0,3)) + (Ellipsis,)
        return sp[slicer]

    # Object arrays: map recursively if the elements are arrays
    if sp.dtype == object:
        out = []
        changed = False
        for x in sp:
            if isinstance(x, np.ndarray):
                out.append(crop_shortest_paths(x))
                changed = True
            else:
                out.append(x)
        if changed:
            return np.array(out, dtype=object)

    print("[warn] Unrecognized shortest_paths shape, leaving unchanged:", sp.shape)
    return sp


# --------- Main processing ---------
def process_maps(maps_arr, origin: str):
    """Return a new list/array with each map cropped to 3x3, preserving original layout."""
    # make iterable of per-map arrays (object arrays or numeric)
    if maps_arr.dtype == object:
        maps_list = list(maps_arr)
    else:
        # assume first dimension is N
        maps_list = [maps_arr[i] for i in range(len(maps_arr))]

    out_list = []
    for m in maps_list:
        tiles12 = _extract_tiles(m, grid_hw=(12,12))
        i0, j0 = compute_origin(origin, (12,12), (3,3))
        tiles3 = crop_tiles(tiles12, i0, j0, 3, 3)  # (3,3,C,th,tw)
        out_list.append(_reformat_like(m, tiles3))

    # keep dtype=object if original was object, else try to stack
    if maps_arr.dtype == object:
        return np.array(out_list, dtype=object)
    try:
        return np.stack(out_list, axis=0)
    except Exception:
        # fallback
        return np.array(out_list, dtype=object)

def process_per_map_array(arr, crop_fn):
    """Map a crop function over per-map arrays (supports object arrays or stacked arrays)."""
    if arr.dtype == object:
        return np.array([crop_fn(x) for x in arr], dtype=object)
    else:
        maps = [arr[i] for i in range(len(arr))]
        out = [crop_fn(x) for x in maps]
        try:
            return np.stack(out, axis=0)
        except Exception:
            return np.array(out, dtype=object)

def main():
    ap = argparse.ArgumentParser(description="Crop 12x12 Warcraft dataset down to 3x3.")
    ap.add_argument("--in_dir",  type=str, default="warcraft_shortest_path/12x12",
                    help="Directory containing the 12x12 npy files")
    ap.add_argument("--out_dir", type=str, default="warcraft_shortest_path/3x3",
                    help="Output directory for the 3x3 subset")
    ap.add_argument("--origin",  type=str, default="top-left",
                    choices=["top-left", "center", "bottom-right", "tl", "c", "br"],
                    help="Which 3x3 to take from the 12x12 grid")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    splits = ["train", "val", "test"]
    for split in splits:
        maps_p = os.path.join(args.in_dir, f"{split}_maps.npy")
        vw_p   = os.path.join(args.in_dir, f"{split}_vertex_weights.npy")
        sp_p   = os.path.join(args.in_dir, f"{split}_shortest_paths.npy")

        if os.path.exists(maps_p):
            maps = np.load(maps_p, allow_pickle=True)
            maps3 = process_maps(maps, args.origin)
            np.save(os.path.join(args.out_dir, f"{split}_maps.npy"), maps3, allow_pickle=True)
            print(f"[ok] {split}_maps.npy: {maps.shape} -> {maps3.shape}")
        else:
            print(f"[skip] missing {maps_p}")

        if os.path.exists(vw_p):
            vw = np.load(vw_p, allow_pickle=True)
            vw3 = process_per_map_array(vw, crop_vertex_weights)
            np.save(os.path.join(args.out_dir, f"{split}_vertex_weights.npy"), vw3, allow_pickle=True)
            print(f"[ok] {split}_vertex_weights.npy: {vw.shape} -> {vw3.shape}")
        else:
            print(f"[skip] missing {vw_p}")

        if os.path.exists(sp_p):
            sp = np.load(sp_p, allow_pickle=True)
            sp3 = process_per_map_array(sp, crop_shortest_paths)
            np.save(os.path.join(args.out_dir, f"{split}_shortest_paths.npy"), sp3, allow_pickle=True)
            print(f"[ok] {split}_shortest_paths.npy: {sp.shape} -> {sp3.shape}")
        else:
            print(f"[skip] missing {sp_p}")

    # Copy & adapt info.json if present
    info_in = os.path.join(args.in_dir, "info.json")
    info_out = os.path.join(args.out_dir, "info.json")
    if os.path.exists(info_in):
        try:
            with open(info_in, "r") as f:
                info = json.load(f)
        except Exception:
            info = {}
        # Best-effort patches without assuming exact schema
        info["source_dir"] = os.path.relpath(args.in_dir, start=os.path.dirname(args.out_dir))
        info["grid_shape"] = [3, 3]
        info["crop_origin"] = args.origin
        # keep tile size if present; default to 8x8
        if "tile_shape" not in info:
            info["tile_shape"] = [8, 8]
        with open(info_out, "w") as f:
            json.dump(info, f, indent=2)
        print(f"[ok] wrote {info_out}")
    else:
        print(f"[skip] no info.json at {info_in}")

if __name__ == "__main__":
    main()
