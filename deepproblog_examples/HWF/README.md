# HWF (DeepProbLog)

Fetch the handwritten-symbol data into `data/` first (see `data/split_train_val.py`
and the HWF release of Li et al., 2020).

```bash
python hwf_class.py                                             # vanilla DeepProbLog
python hwf_prototypes.py --N 1 --model_type vae --save_path out_hwf
python hwf_prototypes.py --N 1 --model_type vae --save_path out_hwf --inference_only
```

`--N` is the formula length, `--model_type diffusion` swaps the decoder for a
DDPM, `--curriculum` trains over increasing lengths. `--inference_only` loads
the trained model and answers the generative `FinishFormula/2` queries.
