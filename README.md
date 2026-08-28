# Declarative DeepProbLog

A declarative extension for neuro-symbolic languages built on neural predicates.
A neural predicate can be called with its image argument unbound; resolution
then grounds it by sampling and decoding a prototype, so one trained model
answers classification and generation queries alike.

Each top-level directory holds one host system: the vanilla baseline and our
declarative version of it.

| Directory | System | Vanilla | Declarative |
|---|---|---|---|
| `deepproblog_examples/` | DeepProbLog | `mnist_class.py`, `mnistr_class.py`, `hwf_class.py` | `mnist_prototypes.py`, `mnist_n_prototypes.py`, `hwf_prototypes.py` |
| `deepstochlog_examples/` | DeepStochLog | `mathexpression.py`, `run_warcraft_pathfinding.py` | `mathexpression_prototype.py`, `warcraft_vae.py` |
| `neurasp_examples/` | NeurASP | `run_mnist_neurasp.py` | `run_mnist_declarative_neurasp.py` |
| `slash_examples/` | SLASH | `run_mnist_slash.py` | `run_mnist_declarative_slash.py` |
| `deepseaproblog_examples/` | DeepSeaProbLog | `run_mnist_dsp.py` | — |
| `vael_examples/` | VAEL | `run_mnist_vael.py` | — |
| `scallop_examples/` | Scallop | `run_mnist_scallop.py` | — |

Every runner accepts `--help`. See the README in each directory for its
environment and commands.

## Setup

DeepProbLog and DeepStochLog need `deepproblog-dev`
(<https://github.com/ML-KULeuven/deepproblog-dev>), which ships the added
predicates; follow its install guide, then:

```bash
pip install -r requirements.txt
```

The baselines each need their own environment; see their READMEs.
