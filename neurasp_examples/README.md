# NeurASP

[NeurASP](https://github.com/azreasoners/NeurASP) (Yang et al., IJCAI 2020) on
MNIST `digit/2` and `add/3`, adapted from `examples/mnistAdd/`. The framework
core is vendored unmodified in `neurasp/`.

Environment: Python 3.11 with `torch`, `torchvision`, `clingo`, `tqdm`.

## Baseline

Purely discriminative — neural atoms map given images to ASP atoms, and nothing
binds an image-typed variable, so generative queries cannot be formulated.

```bash
python MNIST/run_mnist_neurasp.py --task digit --epochs 3 --seed 0
python MNIST/run_mnist_neurasp.py --task sum2  --epochs 3 --seed 0
```

## Declarative NeurASP (ours)

The neural atom is backed by a prototype scorer (shared with the SLASH port),
trained through NeurASP's semantic loss. An unbound image argument is grounded
by a prior-weighted choice rule; clingo enumerates the stable models consistent
with the query and each selected class is decoded from its prototype. One
trained model answers both classification and generative queries.

```bash
python MNIST/run_mnist_declarative_neurasp.py --task digit    --epochs 8 --seed 0
python MNIST/run_mnist_declarative_neurasp.py --task addition --epochs 8 --seed 0
python MNIST/run_multiadd_query.py --engine neurasp --checkpoint <path> --n-masked 4
```

Batch size 100 is required; per-example updates diverge the decoder.
`run_multiadd_query.py` poses masked `multi_add/9` queries to a trained model.
