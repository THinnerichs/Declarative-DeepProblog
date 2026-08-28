"""NeurASP baseline on MNIST digit/2 (direct supervision) and add/3
(distant supervision, their mnistAdd task).

Adapted from examples/mnistAdd/mnistAdd.py of the official NeurASP
repository (https://github.com/azreasoners/NeurASP, Yang et al.,
IJCAI 2020). NeurASP is purely discriminative: neural atoms map given
inputs to distributions over ASP atoms, and no mechanism binds an
image-typed variable in a query, so generative queries cannot be
formulated or evaluated.

Usage:
    python run_mnist_neurasp.py --task digit --epochs 3 --seed 0
    python run_mnist_neurasp.py --task sum2  --epochs 3 --seed 0
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
import torchvision
from torchvision.transforms import transforms

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent / "neurasp"))
_DATA_ROOT = _HERE.parent.parent / "deepproblog_examples" / "MNIST" / "data"

from neurasp import NeurASP


class Net(nn.Module):
    """Digit network from the official mnistAdd example."""

    def __init__(self):
        super(Net, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
            nn.Softmax(1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 16 * 4 * 4)
        return self.classifier(x)


DPROGRAM_SUM2 = """
img(i1). img(i2).
addition(A,B,N) :- digit(0,A,N1), digit(0,B,N2), N=N1+N2.
nn(digit(1,X), [0,1,2,3,4,5,6,7,8,9]) :- img(X).
"""

DPROGRAM_DIGIT = """
img(i1).
nn(digit(1,X), [0,1,2,3,4,5,6,7,8,9]) :- img(X).
"""

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)


def build_dataset(task, seed, data_size=None):
    train = torchvision.datasets.MNIST(
        root=str(_DATA_ROOT), train=True, download=True, transform=transform
    )
    indices = list(range(len(train)))
    random.Random(seed).shuffle(indices)

    dataList, obsList = [], []
    if task == "sum2":
        pairs = [(indices[2 * i], indices[2 * i + 1]) for i in range(len(indices) // 2)]
        if data_size is not None:
            pairs = pairs[:data_size]
        for i, j in pairs:
            dataList.append(
                {"i1": train[i][0].unsqueeze(0), "i2": train[j][0].unsqueeze(0)}
            )
            obsList.append(f":- not addition(i1, i2, {train[i][1] + train[j][1]}).")
    else:
        if data_size is not None:
            indices = indices[:data_size]
        for i in indices:
            dataList.append({"i1": train[i][0].unsqueeze(0)})
            obsList.append(f":- not digit(0, i1, {train[i][1]}).")
    return list(zip(dataList, obsList))


def evaluate(model):
    test = torchvision.datasets.MNIST(
        root=str(_DATA_ROOT), train=False, transform=transform
    )
    loader = torch.utils.data.DataLoader(test, batch_size=500)
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for imgs, lab in loader:
            probs.append(model(imgs).numpy())
            labels.append(lab.numpy())
    probs = np.concatenate(probs)
    labels = np.concatenate(labels)
    digit_acc = (probs.argmax(-1) == labels).mean()

    # most-probable-sum accuracy on disjoint test pairs
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
    parser.add_argument("--task", choices=["digit", "sum2"], default="sum2")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-size", type=int, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    dataset = build_dataset(args.task, args.seed, args.data_size)
    m = Net()
    dprogram = DPROGRAM_SUM2 if args.task == "sum2" else DPROGRAM_DIGIT
    optimizers = {"digit": torch.optim.Adam(m.parameters(), lr=0.001)}
    obj = NeurASP(dprogram, {"digit": m}, optimizers)

    t0 = time.time()
    obj.learn(dataset, epoch=args.epochs, storeSM=False, bar=True,
              task=f"mnist_{args.task}")
    train_time = time.time() - t0

    digit_acc, sum_acc = evaluate(m)
    print(f"task={args.task} seed={args.seed} digit_acc={digit_acc:.4f} "
          f"sum_acc={sum_acc:.4f} train_time={train_time:.1f}s")

    out = _HERE / "results" / "neurasp_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    with open(out, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["task", "seed", "epochs", "data_size",
                        "digit_acc", "sum_acc", "train_time_s"])
        w.writerow([args.task, args.seed, args.epochs, args.data_size,
                    round(digit_acc, 4), round(sum_acc, 4), round(train_time, 1)])


if __name__ == "__main__":
    main()
