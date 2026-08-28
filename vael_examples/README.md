# VAEL baseline

[VAEL](https://github.com/EleMisi/VAEL) (Misino et al., NeurIPS 2022) on the
2-digit MNIST addition benchmark, evaluated with the metrics of our paper.

The official codebase is expected as a sibling checkout:

```
<...>/DeclDeepProblog/VAEL
<...>/DeclDeepProblog/generative-deepproblog/vael_examples
```

Environment (Python 3.11):

```bash
python3.11 -m venv venv && ./venv/bin/pip install torch torchvision numpy \
    problog pysdd tqdm pandas matplotlib
```

The dataset is built by the VAEL codebase on first use.

```bash
PY=./venv/bin/python ./MNIST/run_all.sh    # 3 seeds, 50 epochs
./venv/bin/python MNIST/aggregate.py       # aggregate the numbers
```

`run_mnist_vael.py` uses VAEL's published configuration, program, loss and
early stopping; only the evaluation is ours. Notes, all stated in the paper:

* The published MNIST config cannot generate (sigmoid decoder + standardized
  images + Laplace loss); the runs use its own `--normalization unit
  --rec-loss BCE` option instead.
* VAEL has no single-digit variant, so `digit/2` is read off the two-digit
  model and `multi_add/9` is not expressible.
* Generated images are labelled with the same 1-NN pixel oracle as every other
  system in the paper.
* Two compatibility patches in `run_mnist_vael.py`, documented there:
  `torch.load(..., weights_only=False)` and a resampling guard around
  `gumbel_softmax`.

`analyze_digit_symmetry.py` quantifies the per-position identifiability problem.
