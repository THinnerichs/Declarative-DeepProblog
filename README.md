# Declarative DeepProblog

Declarative DeepProblog is a declarative extension to DeepProblog, DeepStochlog and all neuro-symbolic languages that rely on neural predicates. 

## Setup:
This problem relies on `deepproblog-dev` (available from `https://github.com/ML-KULeuven/deepproblog-dev`), which provides a complete setup guide.

Further we need to implement the predicates for cosine similarity, mse, distributional cosine similarity, and rbf. To do so add the implementation to the engine and add them.

## Running the experiments:

The experiments are sorted by their respective model, i.e. `DeepProblog` and `DeepStochlog`. 
To run the declarative extension, navigate to any of the examples and run `generative.py`. 
Run `python generative.py --h` to see all available options.

## How to make your NeSy program declarative:
Two things are necessary:
1. Encoder and decoder networks that map your entities into latent space (and back), and
2. The DPL/DSL model formulation. 

