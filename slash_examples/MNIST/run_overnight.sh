#!/bin/zsh
# Overnight SLASH runs: 3 seeds x 30 epochs in parallel (~7-8 h on an
# M-series CPU). Results land in results/addition_seed{0,1,2}/summary.csv
# (discriminative + generative accuracy, computed at the end of each run).
#
# PYBIN must point to a Python 3.11 env with:
#   torch torchvision clingo scikit-learn joblib seaborn "networkx<3"
#   tensorboard pillow matplotlib tqdm
# The env used today (recreate with the pip line above if it is gone):
PYBIN=${PYBIN:-/private/tmp/claude-501/-Users-thinnerichs-Documents-Work-Delft-Projects-NeSy-DeclDeepProblog-generative-deepproblog/a126fce5-3614-4d92-b3a2-5a85b4bce8db/scratchpad/venv-scallop/bin/python}

if ! $PYBIN -c "import torch, clingo" 2>/dev/null; then
  echo "PYBIN ($PYBIN) is missing or lacks torch/clingo — set PYBIN to a"
  echo "python3.11 with the packages listed at the top of this script."
  exit 1
fi

cd "$(dirname "$0")"
mkdir -p results
for seed in 0 1 2; do
  nohup caffeinate -i "$PYBIN" run_mnist_slash.py \
    --epochs 30 --seed "$seed" --p-num 4 \
    > "results/slash_seed${seed}.log" 2>&1 &
done
echo "Started 3 SLASH runs (30 epochs each). Progress:"
echo "  tail -f results/slash_seed0.log | grep epoch"
