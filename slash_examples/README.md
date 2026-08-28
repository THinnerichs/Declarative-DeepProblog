# SLASH

[SLASH](https://github.com/askrix/SLASH) (Skryagin et al., KR 2022 / JAIR 2023)
on MNIST `add/3`. The SLASH core and EinsumNetworks are vendored in
`slash_src/` with three portability patches (CPU fallback for the hard-coded
CUDA device, two numpy>=1.24 fixes), all marked by comments.

Environment: Python 3.11 with `torch`, `torchvision`, `clingo`,
`scikit-learn`, `joblib`, `seaborn`, `networkx<3`, `tensorboard`, `tqdm`,
`matplotlib`.

## Baseline

Adapted from their `mnist_generative` experiment: the neural-probabilistic
predicate is an EinsumNetwork circuit trained by alternating EM likelihood
maximisation with SLASH's ASP distant-supervision step. Sampling is
extra-logical — the class is chosen in Python, not by the reasoner (see
`DECLARATIVITY.md`).

```bash
python MNIST/run_mnist_slash.py --task addition --epochs 10 --seed 0
python MNIST/run_mnist_slash.py --task addition --epochs 10 --seed 0 --no-em   # discriminative only
```

## Declarative SLASH (ours)

The prototype-based declarative neural predicate of this repository ported to
SLASH/ASP, trained end to end through SLASH's semantic loss. Unbound image
arguments are grounded by clingo enumerating the prototype classes and decoding
the selected prototype.

```bash
python MNIST/run_mnist_declarative_slash.py --task digit    --epochs 8 --recon-weight 10 --seed 0
python MNIST/run_mnist_declarative_slash.py --task addition --epochs 8 --recon-weight 10 --seed 0
```

Longer training degrades generation: the class softmax saturates and starves
the decoder.

Results go to `MNIST/results/<run>/summary.csv`, per-epoch accuracies to
`history.csv`, sample grids to `samples_epoch*.png`. Generative accuracy uses
the same 1-nearest-neighbour pixel oracle as the Declarative DeepProbLog
experiments.
