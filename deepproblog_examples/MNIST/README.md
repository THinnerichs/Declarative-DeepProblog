# MNIST and MNIST-R (DeepProbLog)

MNIST downloads itself on first run.

## Vanilla DeepProbLog

```bash
python mnist_class.py                       # digit/2 or add/3 (set `problem` at the top)
python mnistr_class.py --task not34         # MNIST-R: not34 count3 count34 lessthan sum2 sum3 sum4
```

## Declarative DeepProbLog

```bash
python mnist_prototypes.py --problem digit    --model_type vae --save_path out_digit
python mnist_prototypes.py --problem addition --model_type vae --save_path out_add
python mnist_n_prototypes.py --problem digit  --model_type vae --save_path out_digit_3p  # 3 prototypes/class
```

`--model_type diffusion` swaps the decoder for a DDPM. `--problem` also takes
the MNIST-R tasks. Generated images are written to `--save_path`.

Add `--inference_only` to load the model trained previously and run the
inference program (no reconstruction term). Generative queries:

```bash
python mnist_prototypes.py --problem digit --save_path out_digit --inference_only --run_rq3_1
python mnist_prototypes.py --problem digit --save_path out_digit --inference_only --run_rq3_2 --rq3_2_len 4
```

`--run_rq3_1` scores simple generative queries, `--run_rq3_2` the masked
`multi_add` queries. Both compare every generated image against the training
set, so they take a while.
