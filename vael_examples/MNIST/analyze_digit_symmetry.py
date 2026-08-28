"""Is VAEL's per-position digit accuracy identifiable under distant supervision?

VAEL learns two separate digit distributions, one per position of the
two-digit image, from the sum alone. Any pair of relabellings
pi1(a) = a + k, pi2(b) = b - k leaves every sum unchanged, so the digit
assignment is only identified up to such a shift (DeepProbLog does not
have this degeneracy: it uses the same network for both positions).

This script reports, per seed, the raw per-position digit accuracy and
the accuracy after the best shift k, together with the shift that is
actually realized.

Usage:
    python analyze_digit_symmetry.py [--device mps]
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch

import run_mnist_vael as R

_HERE = Path(__file__).parent


@torch.no_grad()
def predictions(model, test_set):
    from utils.graph_semiring import GraphSemiring
    model.eval()
    test_set.reset_counter()
    preds, trues = [], []
    for _ in range(len(test_set)):
        images, labels = test_set[0]
        data = torch.as_tensor(images, dtype=torch.float)[:, None].to(model.device)
        model.semiring = GraphSemiring(data.shape[0], model.device)
        mu, _ = model.encoder(data)
        facts = model.compute_facts_probability(mu[:, :model.latent_dim_sym])
        preds.append(facts.argmax(dim=-1).cpu().numpy())
        trues.append(np.asarray(labels)[:, :2])
    return np.concatenate(preds), np.concatenate(trues)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps")
    parser.add_argument("--normalization", choices=["standard", "unit"],
                        default="unit")
    parser.add_argument("--glob", default="vael_seed*/vael.pt")
    args = parser.parse_args()

    rows = []
    for ckpt in sorted((_HERE / "results").glob(args.glob)):
        seed = ckpt.parent.name.split("seed")[-1]
        model, sets = R.build(torch.device(args.device),
                              normalization=args.normalization)
        model.load_state_dict(torch.load(ckpt, map_location=args.device))
        pred, true = predictions(model, sets["test"])
        raw = float((pred == true).mean())
        best_k, best_acc = 0, -1.0
        for k in range(-9, 10):
            shifted = np.stack([pred[:, 0] + k, pred[:, 1] - k], axis=1)
            acc = float((shifted == true).mean())
            if acc > best_acc:
                best_k, best_acc = k, acc
        sums_ok = float(((pred[:, 0] + pred[:, 1]) == true.sum(axis=1)).mean())
        print(f"seed {seed}: raw {100 * raw:5.1f}%  best-shift(k={best_k:+d}) "
              f"{100 * best_acc:5.1f}%  sum-from-digits {100 * sums_ok:5.1f}%")
        rows.append(dict(seed=seed, raw_digit_acc=raw, best_shift=best_k,
                         shifted_digit_acc=best_acc, sum_from_digits=sums_ok))

    with open(_HERE / "results" / "digit_symmetry.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
