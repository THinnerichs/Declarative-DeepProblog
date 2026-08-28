# Scallop baseline

[Scallop](https://github.com/scallop-lang/scallop) on MNIST `digit/2`
(`identity`) and `add/3` (`sum_2`), adapted from `experiments/mnist/`.
Purely discriminative: relations are populated from given inputs, so generative
queries cannot be formulated.

`scallopy` must be built from source with a nightly toolchain:

```bash
cd scallop/etc/scallopy
RUSTC_BOOTSTRAP=1 maturin build --release -i $(which python)
pip install ../../target/wheels/scallopy-*.whl torch torchvision tqdm
```

On macOS the forward functions need `dispatch="serial"`.

```bash
python MNIST/run_mnist_scallop.py --task digit --n-epochs 3 --seed 0
python MNIST/run_mnist_scallop.py --task sum2  --n-epochs 3 --seed 0
```

Results are appended to `MNIST/results/scallop_results.csv`.
