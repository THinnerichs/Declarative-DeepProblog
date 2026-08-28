import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def adjust_image_per_tile(
    input_path: Path,
    output_path: Path,
    N: int = 3,
    min_factor: float = 0.2,
    max_factor: float = 2.0,
):
    """
    Adjust right-half intensity per tile & per channel to match the left-half tiles.

    Assumes:
      - input image is [ left_map | right_map ]
      - each map is an N x N grid of tiles
    """
    img = Image.open(input_path).convert("RGB")
    arr = np.array(img).astype(np.float32)  # (H,W,3) in [0,255]

    H, W, C = arr.shape
    assert C == 3, f"Expected RGB image, got shape {arr.shape}"

    mid = W // 2
    left = arr[:, :mid, :]      # (H, mid, 3)
    right = arr[:, mid:, :]     # (H, mid, 3)

    # infer tile sizes
    tile_h = H // N
    tile_w = (mid) // N  # width of one tile within a half

    for i in range(N):
        for j in range(N):
            y0 = i * tile_h
            y1 = (i + 1) * tile_h
            x0 = j * tile_w
            x1 = (j + 1) * tile_w

            # left and right tiles
            lt = left[y0:y1, x0:x1, :]   # (tile_h, tile_w, 3)
            rt = right[y0:y1, x0:x1, :]  # (tile_h, tile_w, 3)

            # mean per channel
            lt_mean = lt.reshape(-1, C).mean(axis=0)        # (3,)
            rt_mean = rt.reshape(-1, C).mean(axis=0)        # (3,)
            rt_mean = np.where(rt_mean == 0, 1.0, rt_mean)  # avoid /0

            factors = lt_mean / rt_mean                     # (3,)
            factors = np.clip(factors, min_factor, max_factor)

            # scale right tile per channel
            rt_adj = rt * factors[None, None, :]            # broadcast
            rt_adj = np.clip(rt_adj, 0.0, 255.0)

            right[y0:y1, x0:x1, :] = rt_adj

    combined = np.concatenate([left, right], axis=1).astype(np.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(combined, mode="RGB").save(output_path)
    print(f"Saved adjusted image to {output_path}")


def adjust_epoch(
    dir_path: Path,
    epoch: int,
    out_dir: Path,
    N: int = 3,
    min_factor: float = 0.2,
    max_factor: float = 2.0,
    inplace: bool = False,
):
    """
    Adjust all images for one epoch in a directory.

    Expects filenames like: epoch_XXX_map_YYY.png
    where XXX is zero-padded epoch number.

    If inplace=False, saves as epoch_XXX_map_YYY_adj.png in out_dir.
    """
    pattern = f"epoch_{epoch:03d}_*.png"
    files = sorted(dir_path.glob(pattern))

    if not files:
        print(f"No files found in {dir_path} matching {pattern}")
        return

    print(f"Found {len(files)} files for epoch {epoch:03d} in {dir_path}")

    for f in files:
        if inplace:
            out_path = out_dir / f.name
        else:
            out_path = out_dir / f"{f.stem}_adj{f.suffix}"

        adjust_image_per_tile(
            input_path=f,
            output_path=out_path,
            N=N,
            min_factor=min_factor,
            max_factor=max_factor,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Per-tile, per-channel intensity matching for combined [left|right] images."
    )
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="Directory containing epoch_XXX_map_YYY.png images",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        help="Epoch number to process (e.g. 1 for epoch_001_*.png)",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Optional: single image to process instead of an epoch pattern",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to write adjusted images (default: same as --dir for inplace, or dir/adjusted)",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="If set, overwrite images in-place (otherwise suffix '_adj' is added).",
    )
    parser.add_argument("--N", type=int, default=3, help="Number of tiles per side (default: 3)")
    parser.add_argument("--min_factor", type=float, default=0.2)
    parser.add_argument("--max_factor", type=float, default=2.0)

    args = parser.parse_args()

    dir_path = Path(args.dir)
    if args.output_dir is None:
        if args.inplace:
            out_dir = dir_path
        else:
            out_dir = dir_path / "adjusted"
    else:
        out_dir = Path(args.output_dir)

    if args.input:
        # Just one image
        input_path = Path(args.input)
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        if args.inplace:
            out_path = input_path
        else:
            out_path = out_dir / f"{input_path.stem}_adj{input_path.suffix}"
        adjust_image_per_tile(
            input_path=input_path,
            output_path=out_path,
            N=args.N,
            min_factor=args.min_factor,
            max_factor=args.max_factor,
        )
    else:
        if args.epoch is None:
            raise ValueError("Either --input or --epoch must be provided.")
        adjust_epoch(
            dir_path=dir_path,
            epoch=args.epoch,
            out_dir=out_dir,
            N=args.N,
            min_factor=args.min_factor,
            max_factor=args.max_factor,
            inplace=args.inplace,
        )


if __name__ == "__main__":
    main()
