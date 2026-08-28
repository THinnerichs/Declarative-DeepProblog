"""Masked multi_add/9 queries against a trained declarative ASP model.

This is the arbitrary-query test for the declarative claim: the same
trained prototype model and the same grounding mechanism used for
digit/2 and add/3 answer a query type never seen in training. Following
the RQ3-2 protocol of the paper, we take two 4-digit numbers (8 test
images), replace 4 randomly chosen images with variables, and query

    multi_add([I1,I2,I3,I4], [I5,I6,I7,I8], S)

with the true sum S. Bound images contribute their posterior from the
prototype NPP; unbound ones a uniform prior choice rule; clingo
enumerates the stable models consistent with the sum; we take the most
probable model and materialize its classes by decoding the prototypes.
A query counts as correct only if the nearest-neighbour label of every
generated image equals the ground-truth label of the masked slot.

Fully unbound multi_add (8 variables, 10^8 worlds) remains intractable
for exact enumeration, as it is for every baseline.

Usage:
    python run_multiadd_query.py --checkpoint results/declarative_seed3/prototype_npp.pt \
        --engine neurasp --n-queries 1000 --seed 3
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
from torchvision.transforms import transforms

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent / "neurasp"))
sys.path.insert(0, str(_HERE.parent.parent / "slash_examples" / "MNIST"))

from mvpp import MVPP  # NeurASP's MVPP (loaded first, wins in sys.modules)
from run_mnist_declarative_slash import PrototypeNPP, NNOracle

_DATA_ROOT = _HERE.parent.parent / "deepproblog_examples" / "MNIST" / "data"

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),
     transforms.Lambda(lambda x: torch.flatten(x))]
)

TERMS = [f"x{i}" for i in range(1, 9)]
_SUM = "S = 1000*D1+100*D2+10*D3+D4+1000*D5+100*D6+10*D7+D8."
# NeurASP ground atoms: digit(0,term,value); SLASH: digit(0,1,term,value)
RULES = {
    "neurasp": ("multi_add(S) :- "
                + ", ".join(f"digit(0,x{i},D{i})" for i in range(1, 9))
                + ", " + _SUM),
    "slash": ("multi_add(S) :- "
              + ", ".join(f"digit(0,1,x{i},D{i})" for i in range(1, 9))
              + ", " + _SUM),
}


def solve_masked(npp, images, labels, masked_idx, topk_bound=2,
                 engine="neurasp"):
    # Bound slots are grounded only over their topk_bound most probable
    # classes (renormalized): symbolic grounding is otherwise over all
    # 10^8 digit combinations regardless of the probabilities — exactly
    # the blow-up that makes the fully unbound query intractable.
    """Return the MAP class assignment for all 8 slots (bound slots via
    NPP posterior, masked slots via uniform prior), constrained by the
    true sum, or None if no stable model exists."""
    true_sum = (1000 * labels[0] + 100 * labels[1] + 10 * labels[2] + labels[3]
                + 1000 * labels[4] + 100 * labels[5] + 10 * labels[6] + labels[7])
    with torch.no_grad():
        posts = npp(torch.stack(images)).numpy()  # [8, 10]
    lines, value_lists, prob_lists = [], [], []
    for k, term in enumerate(TERMS):
        if k in masked_idx:
            values, probs = list(range(10)), [0.1] * 10
        else:
            ranked = sorted(enumerate(posts[k].tolist()),
                            key=lambda kv: -kv[1])[:topk_bound]
            z = sum(p for _, p in ranked)
            values = [v for v, _ in ranked]
            probs = [p / z for _, p in ranked]
        value_lists.append(values)
        prob_lists.append(probs)
        atom = (f"digit(0,1,{term}," if engine == "slash"
                else f"digit(0,{term},")
        lines.append("; ".join(f"@{max(p, 1e-6):.6f} {atom}{v})"
                               for v, p in zip(values, probs)) + ".")
    program = "\n".join(lines) + "\n" + RULES[engine] + "\n"
    query = f":- not multi_add({true_sum})."
    if engine == "slash":
        from slash_mvpp_loader import SlashMVPP
        models = SlashMVPP(program).find_all_SM_under_query(query)
        if len(models) == 0:
            return None
        # SLASH models are atom-string lists; map values to per-term indices
        rows = []
        for model in models:
            row = [None] * 8
            for a in model:
                if a.startswith("digit(0,1,"):
                    term, val = a[len("digit(0,1,"):-1].rsplit(",", 1)
                    if term in TERMS:
                        row[TERMS.index(term)] = value_lists[
                            TERMS.index(term)].index(int(val))
            if None not in row:
                rows.append(row)
        if not rows:
            return None
        model_arr = np.asarray(rows)
    else:
        models = MVPP(program).find_k_SM_under_obs(query, k=0)
        if len(models) == 0:
            return None
        # models hold per-term indices into each term's grounded value list
        model_arr = np.asarray(models)  # [n, 8], pc order = TERMS order
    scores = np.array([
        sum(np.log(prob_lists[k][row[k]]) for k in range(8))
        for row in model_arr
    ])
    best = model_arr[scores.argmax()]
    return [value_lists[k][best[k]] for k in range(8)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n-queries", type=int, default=1000)
    parser.add_argument("--n-masked", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples-per-slot", type=int, default=1)
    parser.add_argument("--topk-bound", type=int, default=2)
    parser.add_argument("--engine", choices=["neurasp", "slash"],
                        default="neurasp")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    npp = PrototypeNPP(recon_weight=10.0)
    npp.load_state_dict(torch.load(args.checkpoint))
    npp.eval()
    oracle = NNOracle()

    test = torchvision.datasets.MNIST(root=str(_DATA_ROOT), train=False,
                                      transform=transform)
    indices = list(range(len(test)))
    random.shuffle(indices)

    correct = total = no_model = 0
    gen_batch, meta = [], []
    for q in range(args.n_queries):
        idxs = indices[8 * q: 8 * q + 8]
        if len(idxs) < 8:
            break
        images = [test[i][0] for i in idxs]
        labels = [test[i][1] for i in idxs]
        masked = sorted(random.sample(range(8), args.n_masked))
        assignment = solve_masked(npp, images, labels, masked,
                                  topk_bound=args.topk_bound,
                                  engine=args.engine)
        total += 1
        if assignment is None:
            no_model += 1
            continue
        for k in masked:
            img = npp.generate(int(assignment[k]), args.samples_per_slot).numpy()
            gen_batch.append(img[0])
            meta.append((q, labels[k]))
    # nearest-neighbour check, batched
    nn_labels = oracle.labels(np.stack(gen_batch)) if gen_batch else np.array([])
    ok_per_query = {}
    for (q, true_label), nn in zip(meta, nn_labels):
        ok_per_query.setdefault(q, True)
        if nn != true_label:
            ok_per_query[q] = False
    answered = len(ok_per_query)
    correct = sum(ok_per_query.values())
    print(f"checkpoint={args.checkpoint}")
    print(f"queries={total} answered={answered} no_model={no_model} "
          f"correct={correct} acc_over_all={correct / total:.4f}")


if __name__ == "__main__":
    main()
