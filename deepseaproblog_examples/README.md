# DeepSeaProbLog baseline

[DeepSeaProbLog](https://github.com/ML-KULeuven/deepseaproblog) (De Smet et al.,
UAI 2023) on MNIST `digit/2` and `add/3`. The framework is vendored unmodified
in `deepseaproblog/`; `MNIST/models/mnist_generation.pl` is adapted from their
LOGICVAE example. Its generative direction is explicitly programmed and trained
(a dedicated auto-encoding objective per query mode), so each query mode needs
its own program and training run.

Environment: Python 3.11 with `tensorflow==2.15.1`,
`tensorflow-probability==0.23.0`, `problog`, `pysdd`, `torch`, `torchvision`,
`matplotlib`.

```bash
python MNIST/run_mnist_dsp.py --task digit    --epochs 3 --seed 0
python MNIST/run_mnist_dsp.py --task addition --epochs 3 --seed 0
```

Results are written to `MNIST/results/<task>_seed<seed>/summary.csv`.
Generative accuracy uses the same 1-nearest-neighbour pixel oracle as the
Declarative DeepProbLog experiments.
