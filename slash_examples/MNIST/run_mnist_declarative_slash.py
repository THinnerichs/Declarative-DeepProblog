"""Declarative SLASH: prototype-based neural predicates in ASP.

This ports the declarative-DeepProbLog construction (prototype_vae.pl)
to SLASH/ASP. The neural-probabilistic predicate digit/2 is *defined
generatively* through an encoder/decoder pair and one learnable Gaussian
prototype per class:

    score(x, c) = N(enc(x); mu_c, sigma_c) * (1 - MSE(x, dec(z_c))),
    z_c ~ N(mu_c, sigma_c),   P(c | x) = score(x, c) / sum_c' score(x, c')

exactly mirroring encode/decode in models/prototype_vae.pl. This class
distribution is plugged into SLASH as the NPP output and trained end to
end through SLASH's ASP semantic loss under distant supervision
(:- not addition(i1, i2, S)) — no direct labels, no separate generative
training phase.

Because the prototypes discretize the image domain, an *unbound* image
argument can be grounded by resolution: its npp atom is grounded with
the prototype prior over the 10 classes, clingo enumerates the stable
models consistent with the query, and each model's class assignment is
materialized as an image by decoding a sample of that prototype. The
same trained model thus answers all query modes:

    classification   digit(img, ?), addition(img1, img2, ?)
    generation       digit(X, 7), addition(X, Y, 13)   [X, Y unbound]

Usage:
    python run_mnist_declarative_slash.py --epochs 10 --seed 0
"""
import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import transforms

_HERE = Path(__file__).parent
_SRC = _HERE.parent / "slash_src"
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_SRC / "SLASH"))

from mvpp import MVPP
from slash import SLASH

_DATA_ROOT = _HERE.parent.parent / "deepproblog_examples" / "MNIST" / "data"

LATENT_DIM = 12

# [-1, 1] pixel space, as in the Declarative DeepProbLog experiments
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),
     transforms.Lambda(lambda x: torch.flatten(x))]
)

TRAIN_PROGRAM = """
img(i1). img(i2).
addition(A,B,N):- digit(0,+A,-N1), digit(0,+B,-N2), N=N1+N2.
npp(digit(1,X), [0,1,2,3,4,5,6,7,8,9]) :- img(X).
"""

# direct supervision on single digits
TRAIN_PROGRAM_DIGIT = """
img(i1).
digitpred(N) :- digit(0,+A,-N), img(A).
npp(digit(1,X), [0,1,2,3,4,5,6,7,8,9]) :- img(X).
"""


class PrototypeNPP(nn.Module):
    """Prototype-based neural predicate (cf. models/prototype_vae.pl):
    encoder + decoder + one Gaussian prototype per class. forward()
    returns the normalized class scores; generate() decodes prototype
    samples for unbound arguments."""

    def __init__(self, latent_dim=LATENT_DIM, recon_weight=1.0,
                 decoder_arch="mlp"):
        super().__init__()
        self.latent_dim = latent_dim
        # temperature on the image-similarity factor: the latent-likelihood
        # term spans tens of log-units while log(1-MSE) spans ~0.2, so
        # without upweighting the decoder receives almost no gradient
        self.recon_weight = recon_weight
        self.decoder_arch = decoder_arch
        # encoder: same shape as the Decl. DPL VAE encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, latent_dim), nn.Tanh(),
        )
        if decoder_arch == "conv":
            # transposed-conv decoder mirroring the Decl. DPL VAE decoder
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 64 * 7 * 7), nn.ReLU(),
                nn.Unflatten(1, (64, 7, 7)),
                nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1,
                                   output_padding=1), nn.ReLU(),
                nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1,
                                   output_padding=1), nn.ReLU(),
                nn.ConvTranspose2d(32, 1, 3, stride=1, padding=1), nn.Tanh(),
                nn.Flatten(),
            )
        else:
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 128), nn.ReLU(),
                nn.Linear(128, 256), nn.ReLU(),
                nn.Linear(256, 784), nn.Tanh(),
            )
        # per class: [mu (latent_dim), logvar (latent_dim)]
        self.prototypes = nn.Embedding(10, 2 * latent_dim)

    def proto_params(self):
        p = self.prototypes.weight  # [10, 2d]
        mu, logvar = p[:, : self.latent_dim], p[:, self.latent_dim:]
        return mu, logvar.clamp(-6.0, 2.0)

    def sample_protos(self):
        mu, logvar = self.proto_params()
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)  # [10, d]

    def forward(self, x, marg_idx=None, type=1):
        """x: [bs, 784] in [-1, 1] -> [bs, 10] class probabilities.
        The marg_idx/type kwargs mirror the EiNet NPP interface; the
        program only uses the posterior direction (type 1)."""
        bs = x.shape[0]
        z = self.encoder(x.view(bs, 1, 28, 28))          # [bs, d]
        mu, logvar = self.proto_params()
        std = (0.5 * logvar).exp()                        # [10, d]
        # log-likelihood of z under each prototype (unnormalized Gaussian,
        # as builtins.likelihood)
        diff = (z.unsqueeze(1) - mu.unsqueeze(0)) / std.unsqueeze(0)
        log_lat = -0.5 * (diff ** 2).sum(-1)              # [bs, 10]
        # reconstruction: decode one fresh prototype sample per image and
        # class (one per grounding, as in the Decl. DPL engine), compare
        # in [0,1] (as builtins.mse)
        z_s = mu.unsqueeze(0) + std.unsqueeze(0) * torch.randn(
            bs, 10, self.latent_dim, device=x.device)
        recon = self.decoder(z_s.reshape(-1, self.latent_dim)).view(bs, 10, 784)
        x01 = (x + 1.0) / 2.0
        r01 = (recon + 1.0) / 2.0
        mse = ((x01.unsqueeze(1) - r01) ** 2).mean(-1)    # [bs,10]
        im_sim = (1.0 - mse).clamp_min(1e-6)
        log_score = log_lat + self.recon_weight * im_sim.log()
        return F.softmax(log_score, dim=-1)

    @torch.no_grad()
    def generate(self, class_idx, n=1):
        """Decode n samples of prototype class_idx -> [n, 784] in [-1,1]."""
        mu, logvar = self.proto_params()
        std = (0.5 * logvar).exp()
        z = mu[class_idx] + std[class_idx] * torch.randn(n, self.latent_dim)
        return self.decoder(z)


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, seed, data_size=None):
        indices = list(range(len(dataset)))
        random.Random(seed).shuffle(indices)
        self.pairs = [(indices[2 * i], indices[2 * i + 1])
                      for i in range(len(indices) // 2)]
        if data_size is not None:
            self.pairs = self.pairs[:data_size]
        self.dataset = dataset

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        i, j = self.pairs[index]
        label = self.dataset[i][1] + self.dataset[j][1]
        return ({"i1": self.dataset[i][0], "i2": self.dataset[j][0]},
                ":- not addition(i1, i2, {}).".format(label))


class DigitDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, seed, data_size=None):
        indices = list(range(len(dataset)))
        random.Random(seed).shuffle(indices)
        if data_size is not None:
            indices = indices[:data_size]
        self.indices = indices
        self.dataset = dataset

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        i = self.indices[index]
        return ({"i1": self.dataset[i][0]},
                ":- not digitpred({}).".format(self.dataset[i][1]))


class NNOracle:
    def __init__(self):
        raw = torchvision.datasets.MNIST(root=str(_DATA_ROOT), train=True)
        X = raw.data.numpy().astype("float32") / 255.0
        self.X = ((X - 0.5) / 0.5).reshape(len(X), -1)
        self.y = raw.targets.numpy()
        self.sq_norms = (self.X ** 2).sum(axis=1)

    def labels(self, generated):
        d = self.sq_norms[None, :] - 2.0 * generated @ self.X.T
        return self.y[d.argmin(axis=1)]


# ---------- declarative query answering ----------

def _prob_rule(term, probs):
    return "; ".join(f"@{p:.6f} digit(0,1,{term},{v})"
                     for v, p in enumerate(probs)) + "."


def answer_generative_query(rule, query, unbound_terms, bound=None):
    """Ground each unbound image term's npp atom with the prototype prior,
    let clingo enumerate the stable models consistent with the query, and
    return per model the class assignment of every unbound term."""
    lines = []
    for t in unbound_terms:
        lines.append(_prob_rule(t, [0.1] * 10))
    if bound:
        for t, probs in bound.items():
            lines.append(_prob_rule(t, probs))
    program = "\n".join(lines) + "\n" + rule + "\n"
    mvpp = MVPP(program)
    models = mvpp.find_all_SM_under_query(query)
    assignments = []
    for model in models:
        assign = {}
        for atom in model:
            if atom.startswith("digit(0,1,"):
                inner = atom[len("digit(0,1,"):-1]
                term, val = inner.rsplit(",", 1)
                if term in unbound_terms:
                    assign[term] = int(val)
        if len(assign) == len(unbound_terms):
            assignments.append(assign)
    return assignments


def quick_gen_digit(npp, oracle, per_class=20):
    """Cheap per-epoch tracker: class-conditional samples vs 1-NN oracle."""
    npp.eval()
    with torch.no_grad():
        gens = np.concatenate([npp.generate(d, per_class).numpy()
                               for d in range(10)])
    npp.train()
    nn_lab = oracle.labels(gens)
    truth = np.repeat(np.arange(10), per_class)
    return float((nn_lab == truth).mean())


def generative_digit_accuracy(npp, oracle, out_dir, samples_per_model=10):
    rule = "digitpred(N) :- digit(0,1,x1,N)."
    rows, correct, total = [], 0, 0
    for d in range(10):
        models = answer_generative_query(rule, f":- not digitpred({d}).", ["x1"])
        for assign in models:
            imgs = npp.generate(assign["x1"], samples_per_model).numpy()
            nn_lab = oracle.labels(imgs)
            correct += (nn_lab == d).sum()
            total += len(nn_lab)
            rows += [[d, assign["x1"], int(l)] for l in nn_lab]
    with open(out_dir / "generative_digit_details.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "asp_class", "nn_label"])
        w.writerows(rows)
    return correct / total


def generative_addition_accuracy(npp, oracle, out_dir, samples_per_model=10):
    rule = ("addition(N) :- digit(0,1,x1,D1), digit(0,1,x2,D2), N=D1+D2.")
    rows, correct, total = [], 0, 0
    for s in range(19):
        models = answer_generative_query(rule, f":- not addition({s}).",
                                         ["x1", "x2"])
        for assign in models:
            i1 = npp.generate(assign["x1"], samples_per_model).numpy()
            i2 = npp.generate(assign["x2"], samples_per_model).numpy()
            n1, n2 = oracle.labels(i1), oracle.labels(i2)
            correct += (n1 + n2 == s).sum()
            total += len(n1)
            rows += [[s, assign["x1"], assign["x2"], int(a), int(b)]
                     for a, b in zip(n1, n2)]
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


def save_sample_grid(npp, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = np.concatenate(
        [npp.generate(d, 5).numpy().reshape(-1, 28) for d in range(10)], axis=1)
    plt.figure(figsize=(20, 10))
    plt.imshow(grid, cmap="gray")
    plt.axis("off")
    plt.savefig(path, bbox_inches="tight", dpi=80)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--p-num", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--recon-weight", type=float, default=1.0)
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
        + (f"_rw{args.recon_weight:g}" if args.recon_weight != 1.0 else "")
        + (f"_{args.decoder}" if args.decoder != "mlp" else "")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    npp = PrototypeNPP(recon_weight=args.recon_weight,
                       decoder_arch=args.decoder)
    program = TRAIN_PROGRAM_DIGIT if args.task == "digit" else TRAIN_PROGRAM
    slash_obj = SLASH(program, {"digit": npp},
                      {"digit": torch.optim.Adam(npp.parameters(), lr=args.lr)})

    train = torchvision.datasets.MNIST(root=str(_DATA_ROOT), train=True,
                                       download=True, transform=transform)
    dataset_cls = DigitDataset if args.task == "digit" else PairDataset
    loader = torch.utils.data.DataLoader(
        dataset_cls(train, args.seed, args.data_size), shuffle=True,
        batch_size=args.batch_size, num_workers=0,
    )

    oracle = NNOracle()
    t0 = time.time()
    history = []
    for e in range(args.epochs):
        slash_obj.learn(dataset_loader=loader, epoch=1,
                        batchSize=args.batch_size, p_num=args.p_num)
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
        w.writerow(["seed", "epochs", "data_size", "train_time_s",
                    "class_digit_acc", "class_add_acc",
                    "gen_digit_acc", "gen_add_acc"])
        w.writerow([args.seed, args.epochs, args.data_size,
                    round(train_time, 1), digit_acc, sum_acc,
                    gen_digit, gen_add])
    with open(out_dir / "history.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "digit_acc", "sum_acc", "quick_gen_digit"])
        w.writerows(history)


if __name__ == "__main__":
    main()
