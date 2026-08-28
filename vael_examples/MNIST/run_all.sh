#!/usr/bin/env bash
# 3 seeds of VAEL on 2-digit MNIST, published configuration and fixed decoder.
set -u
PY="${PY:-python}"
HERE="$(cd "$(dirname "$0")" && pwd)"
for SEED in 0 1 2; do
  # published configuration: Laplace loss on standardized images. Its decoder
  # collapses to a constant zero image (see README); kept for the record.
  "$PY" "$HERE/run_mnist_vael.py" --seed "$SEED" --epochs 50 --n-sample 1000 \
      --out "$HERE/results/published_config_seed${SEED}" \
      > "$HERE/results/published_config_seed${SEED}.log" 2>&1 &
  # same configuration with the codebase's BCE reconstruction option and the
  # images in [0,1], the range its sigmoid decoder can produce
  "$PY" "$HERE/run_mnist_vael.py" --seed "$SEED" --epochs 50 --n-sample 1000 \
      --normalization unit --rec-loss BCE \
      > "$HERE/results/vael_seed${SEED}.log" 2>&1 &
done
wait
echo "done"
