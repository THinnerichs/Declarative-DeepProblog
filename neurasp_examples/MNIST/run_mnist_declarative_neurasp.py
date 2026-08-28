"""Declarative NeurASP: prototype-based neural predicates in ASP.

The same construction as slash_examples/MNIST/run_mnist_declarative_slash.py,
applied to NeurASP (Yang et al., IJCAI 2020): the neural atom digit/2 is
backed by an encoder/decoder pair with one learnable Gaussian prototype
per class,

    P(c | x)  proportional to  N(enc(x); mu_c, sigma_c) * (1 - MSE(x, dec(z_c)))^beta,

trained end to end through NeurASP's semantic loss under distant
supervision (:- not addition(i1, i2, S)) — no direct labels. Because the
prototypes discretize the image domain, an *unbound* image argument is
grounded by a prior-weighted choice rule over its neural atom; clingo
enumerates the stable models consistent with the query, and each model's
class assignment is materialized by decoding a sample of that class's
prototype. One distantly trained model thus answers classification and
generative queries alike.

Usage:
    python run_mnist_declarative_neurasp.py --epochs 8 --seed 0
"""
import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from torchvision.transforms import transforms

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent / "neurasp"))
# reuse the prototype NPP and 1-NN oracle from the Decl-SLASH runner
sys.path.insert(0, str(_HERE.parent.parent / "slash_examples" / "MNIST"))

from neurasp import NeurASP
from mvpp import MVPP
from run_mnist_declarative_slash import (PrototypeNPP, NNOracle,
                                         quick_gen_digit, save_sample_grid)

_DATA_ROOT = _HERE.parent.parent / "deepproblog_examples" / "MNIST" / "data"

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),
     transforms.Lambda(lambda x: torch.flatten(x))]
)

TRAIN_PROGRAM = """
img(i1). img(i2).
addition(A,B,N) :- digit(0,A,N1), digit(0,B,N2), N=N1+N2.
nn(digit(1,X), [0,1,2,3,4,5,6,7,8,9]) :- img(X).
"""

# direct supervision on single digits
TRAIN_PROGRAM_DIGIT = """
img(i1).
digitpred(N) :- digit(0,A,N), img(A).
nn(digit(1,X), [0,1,2,3,4,5,6,7,8,9]) :- img(X).
"""


def build_digit_dataset(seed, data_size=None, batch_size=100):
    """Batched direct-supervision entries: [bs, 784] image tensor for i1
    plus one digit observation per row."""
    train = torchvision.datasets.MNIST(root=str(_DATA_ROOT), train=True,
                                       download=True, transform=transform)
    indices = list(range(len(train)))
    random.Random(seed).shuffle(indices)
    if data_size is not None:
        indices = indices[:data_size]
    dataset = []
    for start in range(0, len(indices) - batch_size + 1, batch_size):
        chunk = indices[start:start + batch_size]
        i1 = torch.stack([train[i][0] for i in chunk])
        obs = [":- not digitpred({}).".format(train[i][1]) for i in chunk]
        dataset.append(({"i1": i1}, obs))
    return dataset


def build_pair_dataset(seed, data_size=None, batch_size=100):
    """Batched NeurASP dataset entries: each entry maps the input terms to
    a [batch_size, 784] tensor and carries one observation per row, which
    NeurASP's learn() processes as a mini-batch (matching the batched
    gradient dynamics of the SLASH port)."""
    train = torchvision.datasets.MNIST(root=str(_DATA_ROOT), train=True,
                                       download=True, transform=transform)
    indices = list(range(len(train)))
    random.Random(seed).shuffle(indices)
    pairs = [(indices[2 * i], indices[2 * i + 1])
             for i in range(len(indices) // 2)]
    if data_size is not None:
        pairs = pairs[:data_size]
    dataset = []
    for start in range(0, len(pairs) - batch_size + 1, batch_size):
        chunk = pairs[start:start + batch_size]
        i1 = torch.stack([train[i][0] for i, _ in chunk])
        i2 = torch.stack([train[j][0] for _, j in chunk])
        obs = [":- not addition(i1, i2, {}).".format(train[i][1] + train[j][1])
               for i, j in chunk]
        dataset.append(({"i1": i1, "i2": i2}, obs))
    return dataset


# ---------- declarative query answering ----------

def _prob_rule(term, probs):
    return "; ".join(f"@{p:.6f} digit(0,{term},{v})"
                     for v, p in enumerate(probs)) + "."


def answer_generative_query(rule, query, unbound_terms):
    """Ground each unbound image term's neural atom with the prototype
    prior; clingo enumerates the stable models consistent with the query.
    Returns one class-assignment tuple per stable model, columns in the
    order of unbound_terms."""
    lines = [_prob_rule(t, [0.1] * 10) for t in unbound_terms]
    program = "\n".join(lines) + "\n" + rule + "\n"
    mvpp = MVPP(program)
    models = mvpp.find_k_SM_under_obs(query, k=0)
    return [tuple(int(v) for v in row) for row in models]


def generative_digit_accuracy(npp, oracle, out_dir, samples_per_model=10):
    rule = "digitpred(N) :- digit(0,x1,N)."
    rows, correct, total = [], 0, 0
    for d in range(10):
        models = answer_generative_query(rule, f":- not digitpred({d}).",
                                         ["x1"])
        for (c,) in models:
            imgs = npp.generate(c, samples_per_model).numpy()
            nn_lab = oracle.labels(imgs)
            correct += (nn_lab == d).sum()
            total += len(nn_lab)
            rows += [[d, c, int(l)] for l in nn_lab]
    with open(out_dir / "generative_digit_details.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "asp_class", "nn_label"])
        w.writerows(rows)
    return correct / total


def generative_addition_accuracy(npp, oracle, out_dir, samples_per_model=10):
    rule = "addition(N) :- digit(0,x1,D1), digit(0,x2,D2), N=D1+D2."
    rows, correct, total = [], 0, 0
    for s in range(19):
        models = answer_generative_query(rule, f":- not addition({s}).",
                                         ["x1", "x2"])
        for (c1, c2) in models:
            i1 = npp.generate(c1, samples_per_model).numpy()
            i2 = npp.generate(c2, samples_per_model).numpy()
            n1, n2 = oracle.labels(i1), oracle.labels(i2)
            correct += (n1 + n2 == s).sum()
            total += len(n1)
            rows += [[s, c1, c2, int(a), int(b)] for a, b in zip(n1, n2)]
    with open(out_dir / "generative_addition_details.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sum", "asp_class_1", "asp_class_2",
                    "nn_label_1", "nn_label_2"])
        w.writerows(rows)
    return correct / total


def discriminative_eval(npp):
    test = torchvision.datasets.MNIST(root=str(_DATA_ROOT), train=False,
                                      transform=transform)
    loader = torch.utils.data.DataLoader(test, batch_size=250)
    npp.eval()
    probs, labels = [], []
    with torch.no_grad():
        for imgs, lab in loader:
            probs.append(npp(imgs).numpy())
            labels.append(lab.numpy())
    npp.train()
    probs = np.concatenate(probs)
    labels = np.concatenate(labels)
    digit_acc = (probs.argmax(-1) == labels).mean()
    n_pairs = len(labels) // 2
    p1, p2 = probs[0::2][:n_pairs], probs[1::2][:n_pairs]
    sum_probs = np.zeros((n_pairs, 19))
    for a in range(10):
        for b in range(10):
            sum_probs[:, a + b] += p1[:, a] * p2[:, b]
    true = labels[0::2][:n_pairs] + labels[1::2][:n_pairs]
    sum_acc = (sum_probs.argmax(-1) == true).mean()
    return digit_acc, sum_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--recon-weight", type=float, default=10.0)
    parser.add_argument("--decoder", choices=["mlp", "conv"], default="mlp")
    parser.add_argument("--task", choices=["addition", "digit"],
                        default="addition")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    suffix = "_digit" if args.task == "digit" else ""
    out_dir = _HERE / "results" / (
        f"declarative{suffix}_seed{args.seed}"
        + (f"_{args.decoder}" if args.decoder != "mlp" else "")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    npp = PrototypeNPP(recon_weight=args.recon_weight,
                       decoder_arch=args.decoder)
    program = TRAIN_PROGRAM_DIGIT if args.task == "digit" else TRAIN_PROGRAM
    obj = NeurASP(program, {"digit": npp},
                  {"digit": torch.optim.Adam(npp.parameters(), lr=args.lr)})

    if args.task == "digit":
        dataset = build_digit_dataset(args.seed, args.data_size)
    else:
        dataset = build_pair_dataset(args.seed, args.data_size)

    oracle = NNOracle()
    t0 = time.time()
    history = []
    for e in range(args.epochs):
        obj.learn(dataset, epoch=1, storeSM=False, bar=True,
                  task="mnist_decl")
        digit_acc, sum_acc = discriminative_eval(npp)
        qgen = quick_gen_digit(npp, oracle)
        print(f"epoch {e + 1}: digit_acc={digit_acc:.4f} sum_acc={sum_acc:.4f} "
              f"qgen={qgen:.4f}", flush=True)
        history.append([e + 1, digit_acc, sum_acc, qgen])
        save_sample_grid(npp, out_dir / f"samples_epoch{e + 1}.png")
    train_time = time.time() - t0

    npp.eval()
    digit_acc, sum_acc = history[-1][1], history[-1][2]
    gen_digit = generative_digit_accuracy(npp, oracle, out_dir)
    gen_add = generative_addition_accuracy(npp, oracle, out_dir)
    print(f"final: digit_acc={digit_acc:.4f} sum_acc={sum_acc:.4f} "
          f"gen_digit={gen_digit:.4f} gen_add={gen_add:.4f} "
          f"train_time={train_time:.1f}s", flush=True)

    torch.save(npp.state_dict(), out_dir / "prototype_npp.pt")
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "epochs", "data_size", "recon_weight",
                    "train_time_s", "class_digit_acc", "class_add_acc",
                    "gen_digit_acc", "gen_add_acc"])
        w.writerow([args.seed, args.epochs, args.data_size,
                    args.recon_weight, round(train_time, 1),
                    digit_acc, sum_acc, gen_digit, gen_add])
    with open(out_dir / "history.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "digit_acc", "sum_acc", "quick_gen_digit"])
        w.writerows(history)


if __name__ == "__main__":
    main()
