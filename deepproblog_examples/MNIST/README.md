# MNIST addition experiments

Run 
```bash
python distr_generative.py --save_path "distr_VARX_CONST" --show_all --problem digit --model_type vae
```
for digit classification and the results to RQ1.

For RQ2 run the distant supervision task using:
```bash

python distr_generative.py --save_path "distr_ADD_X_Y" --show_all --problem addition --model_type vae
```
```bash
python digit_class.py
```
will run the vanilla DeepProblog program for these tasks.
Our declarative extension takes approximately 2-3 times as long as the vanilla version. 

For inference add the flag `--inference_only`, which will 1. load a model you trained previously, and 2. will use the inference version of the declarative DeepProblog program (i.e. without reconstruction error for images).

Both commands will automatically generate images and write them to the `save_path` directory.

To get the results for RQ3, use `--inference_only` and set the two booleans `run_RQ3_1` and `run_RQ3_2` to `True`. 
As all generated images have to be compared with all training images, this may take a while.

