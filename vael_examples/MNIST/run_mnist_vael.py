"""VAEL (Misino et al., NeurIPS 2022) on the MNIST addition benchmark.

Runs the official 2-digit MNIST experiment of the VAEL codebase
(https://github.com/EleMisi/VAEL, vendored as ../../../VAEL) with its
published hyper-parameters, and evaluates it with *our* metrics so that
the numbers are comparable with Table 1 of the paper:

  * classification, digit/2   per-position digit accuracy of the
                              distantly trained model (VAEL has no
                              single-digit variant: its program, its
                              decoder and its dataset are defined over
                              the two-digit image)
  * classification, add/3     accuracy of the addition query, i.e. VAEL's
                              own `discriminative_ability`
  * generative,     digit/2   sample z, condition the program on the
                              evidence digit(img,1,d), sample a world,
                              decode, and check the left half of the
                              generated image
  * generative,     add/3     VAEL's own `generative_ability`: condition
                              on evidence addition(img,N), decode, and
                              check that the two halves sum to N

Generated images are labelled with the same nearest-neighbour pixel
oracle used for all other systems in the paper (1-NN over the MNIST
training images), not with VAEL's own MNIST classifier, so that the
generative columns measure the same thing everywhere.

`multi_add/9` is not attempted: VAEL's herbrand base, worlds-queries
matrix and decoder are all built for exactly two digits, so the query
would require a different model trained on a different dataset.

Usage:
    python run_mnist_vael.py --seed 0 --epochs 50
"""
import argparse
import csv
import json
import random
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch import optim

_HERE = Path(__file__).parent
_VAEL = _HERE.parent.parent.parent / "VAEL"
sys.path.insert(0, str(_VAEL))

# the VAEL dataset is a pickled dict of numpy arrays; torch >= 2.6 defaults
# to weights_only=True, which cannot read it
_torch_load = torch.load
torch.load = lambda *a, **kw: _torch_load(*a, **{**kw, "weights_only": False})

# VAEL samples a world with F.gumbel_softmax(log P(w), tau=1, hard=True).
# The Gumbel noise is -log(-log(u)); when the uniform sample underflows to
# exactly 0 or 1 the noise is +-inf and the sampled world becomes NaN, which
# kills the run a few epochs in (observed on the MPS backend). We resample in
# that case, and fall back to the same estimator with the uniform clamped away
# from the boundary. Whenever the stock call is finite -- i.e. always, except
# for the degenerate draw -- this is exactly torch's gumbel_softmax.
_gumbel_softmax = torch.nn.functional.gumbel_softmax


def _safe_gumbel_softmax(logits, tau=1.0, hard=False, eps=1e-10, dim=-1):
    for _ in range(10):
        out = _gumbel_softmax(logits, tau=tau, hard=hard, dim=dim)
        if torch.isfinite(out).all():
            return out
    u = torch.rand_like(logits).clamp(min=1e-7, max=1.0 - 1e-7)
    y_soft = torch.softmax((logits - torch.log(-torch.log(u))) / tau, dim=dim)
    if not hard:
        return y_soft
    index = y_soft.argmax(dim, keepdim=True)
    y_hard = torch.zeros_like(y_soft).scatter_(dim, index, 1.0)
    return (y_hard - y_soft).detach() + y_soft


torch.nn.functional.gumbel_softmax = _safe_gumbel_softmax

from models.vael import MNISTPairsVAELModel  # noqa: E402
from models.vael_networks import (MNISTPairsDecoder, MNISTPairsEncoder,  # noqa: E402
                                  MNISTPairsMLP)
from utils.graph_semiring import GraphSemiring  # noqa: E402
from utils.mnist_utils.mnist_addition_dataset import nMNIST, check_dataset  # noqa: E402
from utils.mnist_utils.problog_model import create_facts  # noqa: E402
from utils.mnist_utils.train import loss_function  # noqa: E402

_MNIST_ROOT = _HERE.parent.parent / "deepproblog_examples" / "MNIST" / "data"

# published configuration (VAEL/config.py: mnist_vael)
CONFIG = dict(latent_dim_sub=8, latent_dim_sym=15, learning_rate=1e-3,
              dropout=0.5, dropout_ENC=0.5, dropout_DEC=0.5,
              recon_w=1e-1, kl_w=1e-5, query_w=1.0, sup_w=0.0,
              rec_loss="LAPLACE", max_epoch=50)
BATCH_SIZE = {"train": 30, "val": 120, "test": 60}
EARLY_STOPPING = dict(patience=20, delta=1e-8)
DATASET_DIM = {"train": 42000, "val": 12000, "test": 6000}
N_DIGITS = 10
SEQ_LEN = 2

RULES = ("addition(X,N) :- digit(X,1,N1), digit(X,2,N2), N is N1 + N2.\n"
         "digits(X,Y):-digit(img,1,X), digit(img,2,Y).")


# --------------------------------------------------------------------------
# evidence programs
# --------------------------------------------------------------------------

def _compile(program):
    from problog.formula import LogicDAG, LogicFormula
    from problog.sdd_formula import SDD
    return SDD.create_from(LogicDAG.create_from(LogicFormula.create_from(program)))


def build_evidence_dict(n_digits=N_DIGITS):
    """Pre-compiled ProbLog programs, one per generative query.

    'addition': the evidence programs of the VAEL codebase, one per sum.
    'digit':    the same construction with the evidence digit(img,1,d),
                which is the single-digit generative query. VAEL ships
                only the addition evidences; the program is written the
                same way, with its own compiled circuit.
    """
    facts = create_facts(SEQ_LEN, n_digits=n_digits)
    head = "".join(f"\n\n% Digit in position {i + 1}\n\n{facts[i]}"
                   for i in range(len(facts)))
    head += "\n\n% Rules\n" + RULES + "\n\n% Digit Query\nquery(digits(X,Y))."
    evidence = {"addition": {}, "digit": {}}
    for s in range((n_digits - 1) * SEQ_LEN + 1):
        evidence["addition"][s] = _compile(
            head + f"\n\n% Addition Evidence\nevidence(addition(img,{s})).")
    for d in range(n_digits):
        evidence["digit"][d] = _compile(
            head + f"\n\n% Digit Evidence\nevidence(digit(img,1,{d})).")
    return evidence


# --------------------------------------------------------------------------
# 1-NN oracle (identical to the one used for the other systems)
# --------------------------------------------------------------------------

class NNOracle:
    """Nearest neighbour in pixel space over the MNIST training images.

    VAEL's decoder ends in a sigmoid, and its own generative metric feeds
    that output straight into an MNIST classifier trained on [0, 1]
    images, so the decoder output is read as an intensity map and mapped
    to the [-1, 1] space this oracle shares with the other runners.
    """

    def __init__(self):
        raw = torchvision.datasets.MNIST(root=str(_MNIST_ROOT), train=True,
                                         download=True)
        X = raw.data.numpy().astype("float32") / 255.0
        self.X = ((X - 0.5) / 0.5).reshape(len(X), -1)
        self.y = raw.targets.numpy()
        self.sq_norms = (self.X ** 2).sum(axis=1)

    def labels(self, generated):
        """generated: [n, 784] sigmoid outputs of the VAEL decoder."""
        x = (generated - 0.5) / 0.5
        d = self.sq_norms[None, :] - 2.0 * x @ self.X.T
        return self.y[d.argmin(axis=1)]


# --------------------------------------------------------------------------
# losses
# --------------------------------------------------------------------------

def digit_batches(dataset, batch_size, seed):
    """Shuffled batches over the same images, for direct supervision.

    VAEL's loader emits one batch per digit pair, because its addition query
    is evaluated once per batch. Direct supervision on the digits has no such
    constraint, and class-homogeneous batches stall it, so we reshuffle the
    same training images across worlds.
    """
    if not hasattr(dataset, "_flat"):
        idx = np.concatenate([v.ravel() for v in dataset.idxs.values()])
        images = ((dataset.images[idx].astype("float32") - dataset.mean)
                  / dataset.std)
        labels = np.concatenate(
            [np.asarray(dataset.labels[idx]), np.full((len(idx), 1), -1)],
            axis=1)
        dataset._flat = (images, labels)
    images, labels = dataset._flat
    order = np.random.default_rng(seed).permutation(len(images))
    for start in range(0, len(order) - batch_size + 1, batch_size):
        sl = order[start:start + batch_size]
        yield images[sl], labels[sl]


def digit_supervision(model, labels):
    """Cross-entropy on both digit distributions of the neural predicate.

    This is direct supervision on digit/2: the world is observed, so the
    addition query term of the VAEL objective is replaced by the label term
    the codebase already defines (its label_cross_entropy), applied to every
    training image instead of to a few. The rest of the objective
    (reconstruction + KL) is unchanged.
    """
    p1 = model.facts_probs[:, 0].gather(1, labels[:, 0:1])
    p2 = model.facts_probs[:, 1].gather(1, labels[:, 1:2])
    pred = torch.cat([p1, p2]).flatten()
    return torch.nn.BCELoss(reduction="mean")(pred, torch.ones_like(pred))


def vael_loss(model, recon, data, mu, logvar, add_prob, labels, task):
    loss, *rest = loss_function(recon, data, mu, logvar, add_prob, model=model,
                                labels=labels, query=(task == "add"),
                                recon_w=CONFIG["recon_w"], kl_w=CONFIG["kl_w"],
                                query_w=CONFIG["query_w"],
                                sup_w=CONFIG["sup_w"], sup=False,
                                rec_loss=CONFIG["rec_loss"])
    if task == "digit":
        loss = loss + CONFIG["query_w"] * digit_supervision(model, labels)
    return loss


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------

@torch.no_grad()
def validation_loss(model, val_set, task="add"):
    """Mean validation ELBO, the quantity VAEL early-stops on."""
    model.eval()
    model.is_train = False
    val_set.reset_counter()
    losses = []
    batches = (digit_batches(val_set, BATCH_SIZE["val"], 0) if task == "digit"
               else (val_set[0] for _ in range(len(val_set))))
    for images, labels in batches:
        data = torch.as_tensor(images, dtype=torch.float)[:, None].to(model.device)
        labels = torch.as_tensor(labels, dtype=torch.long).to(model.device)
        model.semiring = GraphSemiring(data.shape[0], model.device)
        recon, mu, logvar, add_prob = model(data, labels)
        loss = vael_loss(model, recon, data, mu, logvar, add_prob, labels, task)
        losses.append(float(loss))
    return float(np.mean(losses))


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------

@torch.no_grad()
def classification_accuracy(model, test_set):
    """Per-position digit accuracy and addition-query accuracy."""
    model.eval()
    test_set.reset_counter()
    n_digit = n_digit_ok = n_add = n_add_ok = 0
    for _ in range(len(test_set)):
        images, labels = test_set[0]
        data = torch.as_tensor(images, dtype=torch.float)[:, None].to(model.device)
        labels = torch.as_tensor(labels, dtype=torch.long)
        model.semiring = GraphSemiring(data.shape[0], model.device)
        mu, logvar = model.encoder(data)
        facts_probs = model.compute_facts_probability(mu[:, :model.latent_dim_sym])
        pred = facts_probs.argmax(dim=-1).cpu()                    # [bs, 2]
        n_digit_ok += int((pred == labels[:, :2]).sum())
        n_digit += 2 * len(labels)
        queries = list(range(model.mlp.n_facts - 1))               # sums 0..18
        query_prob = torch.stack(
            [model.problog_inference(facts_probs, query=q)[0][:, 0]
             for q in queries], dim=1)
        n_add_ok += int((query_prob.argmax(dim=1).cpu() == labels[:, -2]).sum())
        n_add += len(labels)
    return n_digit_ok / n_digit, n_add_ok / n_add


@torch.no_grad()
def generative_accuracy(model, evidence_dict, oracle, n_sample, out_dir):
    """Generative accuracy for the digit/2 and add/3 queries.

    For every possible query we sample n_sample latent vectors, compute
    P(w | evidence) with the compiled circuit of that query, sample a
    world, decode it, and label the halves of the generated image with
    the 1-NN oracle. A digit query is correct if the generated left half
    is the queried digit; an addition query is correct if both halves
    are correctly generated, i.e. their labels sum to the queried number.
    """
    model.eval()
    model.semiring = GraphSemiring(n_sample, model.device)
    rows, out = [], {}
    for mode in ("digit", "addition"):
        correct = total = 0
        for value, sdd in evidence_dict[mode].items():
            z = torch.randn(n_sample,
                            model.latent_dim_sym + model.latent_dim_sub,
                            device=model.device)
            model.facts_probs = model.compute_facts_probability(
                z[:, :model.latent_dim_sym])
            # body of VAEL's problog_inference_with_evidence, with the
            # compiled circuit passed directly instead of by dict key
            model.update_semiring_weights(model.facts_probs)
            worlds_prob = model.extract_worlds_probability(
                sdd.evaluate(semiring=model.semiring))
            world = torch.nn.functional.gumbel_softmax(
                torch.log(worlds_prob), tau=1, hard=True)
            images = model.decode(z[:, model.latent_dim_sym:],
                                  model.herbrand(world))
            left = images[:, 0, :, :28].reshape(n_sample, -1).cpu().numpy()
            right = images[:, 0, :, 28:].reshape(n_sample, -1).cpu().numpy()
            l_lab, r_lab = oracle.labels(left), oracle.labels(right)
            if mode == "digit":
                ok = l_lab == value
            else:
                ok = (l_lab + r_lab) == value
            correct += int(ok.sum())
            total += n_sample
            rows += [[mode, value, int(a), int(b), bool(o)]
                     for a, b, o in zip(l_lab, r_lab, ok)]
        out[mode] = correct / total
    with open(out_dir / "generative_details.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query", "value", "nn_left", "nn_right", "correct"])
        w.writerows(rows)
    return out["digit"], out["addition"]


# --------------------------------------------------------------------------

def build(device, normalization="standard"):
    """The VAEL model and the three data splits, as configured above.

    normalization="standard" is the codebase's own preprocessing, which
    standardizes the images to zero mean and unit variance (range about
    [-0.4, 2.8]). Its decoder ends in a sigmoid and cannot represent that
    range, and under the Laplace (L1) reconstruction loss every pixel is
    then pushed towards the pixel-wise median, which saturates the decoder
    at 0 (see the README). normalization="unit" keeps the same code path
    but scales the images to [0, 1], the range the sigmoid can produce.
    """
    # ---- data (the dataset of the VAEL codebase) ----
    data_folder = _VAEL / "data" / "MNIST" / f"2mnist_{N_DIGITS}digits"
    data_file = f"2mnist_{N_DIGITS}digits.pt"
    check_dataset(N_DIGITS, str(data_folder), data_file, DATASET_DIM)
    idxs = {split: torch.load(data_folder / f"{split}_indexes.pt")
            for split in ("train", "val", "test")}
    sets = {split: nMNIST(SEQ_LEN, worlds=None, digits=N_DIGITS,
                          batch_size=BATCH_SIZE[split], idxs=idxs[split],
                          mode=split, sup=False, sup_digits=None,
                          data_path=str(data_folder / data_file))
            for split in ("train", "val", "test")}
    if normalization == "unit":
        for s in sets.values():
            s.mean, s.std = 0.0, 255.0

    # ---- model ----
    w_q = torch.zeros(N_DIGITS ** 2, 2 * N_DIGITS)
    for w, (d1, d2) in enumerate(product(range(N_DIGITS), repeat=SEQ_LEN)):
        w_q[w, d1 + d2] = 1.0
    encoder = MNISTPairsEncoder(
        hidden_channels=64,
        latent_dim=CONFIG["latent_dim_sym"] + CONFIG["latent_dim_sub"],
        dropout=CONFIG["dropout_ENC"])
    decoder = MNISTPairsDecoder(label_dim=N_DIGITS * SEQ_LEN,
                                hidden_channels=64,
                                latent_dim=CONFIG["latent_dim_sub"],
                                dropout=CONFIG["dropout_DEC"])
    mlp = MNISTPairsMLP(in_features=CONFIG["latent_dim_sym"],
                        n_facts=N_DIGITS * SEQ_LEN)
    model = MNISTPairsVAELModel(encoder=encoder, decoder=decoder, mlp=mlp,
                                latent_dims=(CONFIG["latent_dim_sym"],
                                             CONFIG["latent_dim_sub"]),
                                model_dict=None, w_q=w_q.to(device),
                                dropout=CONFIG["dropout"], is_train=True,
                                device=device).to(device)
    return model, sets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=CONFIG["max_epoch"])
    parser.add_argument("--n-sample", type=int, default=1000,
                        help="generated images per query value")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--out", default=None)
    parser.add_argument("--limit-batches", type=int, default=None,
                        help="smoke test: batches per epoch")
    parser.add_argument("--task", choices=["add", "digit"], default="add",
                        help="add: distant supervision on the sum, the VAEL "
                             "setup; digit: direct supervision on both digits")
    parser.add_argument("--kl-w", type=float, default=CONFIG["kl_w"],
                        help="weight of the KL term; the published value is "
                             "1e-5, which leaves the style vector unbottlenecked")
    parser.add_argument("--latent-dim-sub", type=int,
                        default=CONFIG["latent_dim_sub"])
    parser.add_argument("--rec-loss", choices=["LAPLACE", "BCE", "MSE"],
                        default=CONFIG["rec_loss"],
                        help="reconstruction loss; the codebase supports all three")
    parser.add_argument("--normalization", choices=["standard", "unit"],
                        default="standard",
                        help="image preprocessing; see build()")
    parser.add_argument("--eval-only", action="store_true",
                        help="evaluate the best checkpoint in --out, no training")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    CONFIG["rec_loss"] = args.rec_loss
    CONFIG["kl_w"] = args.kl_w
    CONFIG["latent_dim_sub"] = args.latent_dim_sub
    device = torch.device(args.device)
    out_dir = Path(args.out or _HERE / "results" / f"vael_seed{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)

    model, sets = build(device, normalization=args.normalization)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

    print("compiling evidence programs ...", flush=True)
    evidence_dict = build_evidence_dict()
    oracle = NNOracle()

    history = []
    best_val, best_epoch, patience = float("inf"), 0, 0
    for epoch in range(1, 0 if args.eval_only else args.epochs + 1):
        model.train()
        model.is_train = True
        sets["train"].reset_counter()
        model.semiring = GraphSemiring(BATCH_SIZE["train"], device)
        t0, losses = time.time(), []
        n_batches = min(len(sets["train"]), args.limit_batches or 10 ** 9)
        diverged = False
        if args.task == "digit":
            batches = digit_batches(sets["train"], BATCH_SIZE["train"],
                                    1000 * args.seed + epoch)
        else:
            batches = (sets["train"][0] for _ in range(len(sets["train"])))
        for _, (images, labels) in zip(range(n_batches), batches):
            data = torch.as_tensor(images, dtype=torch.float)[:, None].to(device)
            labels = torch.as_tensor(labels, dtype=torch.long).to(device)
            optimizer.zero_grad()
            recon, mu, logvar, add_prob = model(data, labels)
            loss = vael_loss(model, recon, data, mu, logvar, add_prob, labels,
                             args.task)
            if not torch.isfinite(loss):
                diverged = True
                break
            loss.backward()
            optimizer.step()
            losses.append(float(loss))
        if diverged:
            # VAEL's ELBO can blow up mid-training (observed on 1 of 5 seeds);
            # the best-validation checkpoint of the epochs before is kept, which
            # is the model the codebase's early stopping would return anyway.
            print(f"training diverged at epoch {epoch}; "
                  f"keeping the checkpoint of epoch {best_epoch}", flush=True)
            break
        model.is_train = False
        val_loss = validation_loss(model, sets["val"], args.task)
        digit_acc, add_acc = classification_accuracy(model, sets["test"])
        history.append(dict(epoch=epoch, loss=float(np.mean(losses)),
                            val_loss=val_loss, digit_acc=digit_acc,
                            add_acc=add_acc, seconds=time.time() - t0))
        print(f"epoch {epoch:3d}  loss {np.mean(losses):10.3f}  "
              f"val {val_loss:10.3f}  digit {digit_acc:.4f}  "
              f"add {add_acc:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        with open(out_dir / "history.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(history[0]))
            w.writeheader()
            w.writerows(history)
        # early stopping on the validation ELBO, as in the VAEL codebase
        if val_loss < best_val - EARLY_STOPPING["delta"]:
            best_val, best_epoch, patience = val_loss, epoch, 0
            torch.save(model.state_dict(), out_dir / "vael.pt")
        else:
            patience += 1
            if patience >= EARLY_STOPPING["patience"]:
                print(f"early stopping at epoch {epoch}", flush=True)
                break

    if args.eval_only and (out_dir / "history.csv").exists():
        # recover which epoch the stored checkpoint came from
        with open(out_dir / "history.csv") as f:
            hist = list(csv.DictReader(f))
        best = min(hist, key=lambda r: float(r["val_loss"]))
        best_epoch, best_val = int(best["epoch"]), float(best["val_loss"])
    model.load_state_dict(torch.load(out_dir / "vael.pt"))
    model.is_train = False
    print(f"best epoch {best_epoch} (val {best_val:.3f})", flush=True)
    digit_acc, add_acc = classification_accuracy(model, sets["test"])

    print("evaluating generative queries ...", flush=True)
    gen_digit, gen_add = generative_accuracy(model, evidence_dict, oracle,
                                             args.n_sample, out_dir)
    summary = dict(seed=args.seed, task=args.task, epochs=args.epochs,
                   best_epoch=best_epoch,
                   class_digit=digit_acc, class_add=add_acc,
                   gen_digit=gen_digit, gen_add=gen_add,
                   n_sample=args.n_sample)
    print(json.dumps(summary, indent=2))
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary))
        w.writeheader()
        w.writerow(summary)


if __name__ == "__main__":
    main()
