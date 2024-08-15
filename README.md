# Declarative DeepProblog

Declarative DeepProblog is a declarative extension to DeepProblog, DeepStochlog and all neuro-symbolic languages that rely on neural predicates. 

## Setup:
This problem relies on `deepproblog-dev` (available from `https://github.com/ML-KULeuven/deepproblog-dev`), which provides a complete setup guide.
An implementation is also provided in the separate `deepproblog.zip`. Please follow the installation instructions.
This also includes the added predicates needed.

Please install the `requirements.txt` afterwards.

## Running the experiments:

The experiments are sorted by their respective model, i.e. `DeepProblog` and `DeepStochlog`. 
To run the declarative extension, navigate to any of the examples and run `distr_generative.py`. 
Run `python distr_generative.py --h` to see all available options.

## How to make your NeSy program declarative:
Two things are necessary:
1. Encoder and decoder networks that map your entities into latent space (and back), and
2. The DPL model formulation. 

