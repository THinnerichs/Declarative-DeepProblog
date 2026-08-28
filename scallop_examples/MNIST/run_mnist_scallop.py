"""Scallop baseline on MNIST digit/2 (direct supervision) and add/3
(distant supervision, sum of two digits).

Adapted from experiments/mnist/sum_2.py and identity.py of the official
Scallop repository (https://github.com/scallop-lang/scallop,
Huang et al., NeurIPS 2021 / Li et al., PLDI 2023). Scallop is purely
discriminative: it has no mechanism for binding images to unbound
variables, so no generative queries can be formulated or evaluated.

Usage:
    python run_mnist_scallop.py --task digit --n-epochs 3 --seed 0
    python run_mnist_scallop.py --task sum2  --n-epochs 3 --seed 0
"""
import csv
import os
import random
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from tqdm import tqdm

import scallopy

_HERE = Path(__file__).parent
_DATA_ROOT = _HERE.parent.parent / "deepproblog_examples" / "MNIST" / "data"

mnist_img_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
])


class MNISTSum2Dataset(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None):
        self.mnist_dataset = torchvision.datasets.MNIST(
            root, train=train, transform=transform, download=True
        )
        self.index_map = list(range(len(self.mnist_dataset)))
        random.shuffle(self.index_map)

    def __len__(self):
        return int(len(self.mnist_dataset) / 2)

    def __getitem__(self, idx):
        (a_img, a_digit) = self.mnist_dataset[self.index_map[idx * 2]]
        (b_img, b_digit) = self.mnist_dataset[self.index_map[idx * 2 + 1]]
        return (a_img, b_img, a_digit + b_digit)

    @staticmethod
    def collate_fn(batch):
        a_imgs = torch.stack([item[0] for item in batch])
        b_imgs = torch.stack([item[1] for item in batch])
        digits = torch.stack([torch.tensor(item[2]).long() for item in batch])
        return ((a_imgs, b_imgs), digits)


class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 2)
        x = F.max_pool2d(self.conv2(x), 2)
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.softmax(x, dim=1)


class MNISTSum2Net(nn.Module):
    def __init__(self, provenance, k):
        super(MNISTSum2Net, self).__init__()
        self.mnist_net = MNISTNet()
        self.scl_ctx = scallopy.ScallopContext(provenance=provenance, k=k)
        self.scl_ctx.add_relation("digit_1", int, input_mapping=list(range(10)))
        self.scl_ctx.add_relation("digit_2", int, input_mapping=list(range(10)))
        self.scl_ctx.add_rule("sum_2(a + b) :- digit_1(a), digit_2(b)")
        self.sum_2 = self.scl_ctx.forward_function(
            "sum_2", output_mapping=[(i,) for i in range(19)], dispatch="serial"
        )

    def forward(self, x):
        (a_imgs, b_imgs) = x
        a_distrs = self.mnist_net(a_imgs)
        b_distrs = self.mnist_net(b_imgs)
        return self.sum_2(digit_1=a_distrs, digit_2=b_distrs)


class MNISTIdentityNet(nn.Module):
    """Direct supervision: the Scallop program is the identity relation
    over the predicted digit (experiments/mnist/identity.py)."""

    def __init__(self, provenance, k):
        super(MNISTIdentityNet, self).__init__()
        self.mnist_net = MNISTNet()
        self.scl_ctx = scallopy.ScallopContext(provenance=provenance, k=k)
        self.scl_ctx.add_relation("digit", int, input_mapping=list(range(10)))
        self.scl_ctx.add_rule("pred(a) :- digit(a)")
        self.pred = self.scl_ctx.forward_function(
            "pred", output_mapping=[(i,) for i in range(10)], dispatch="serial"
        )

    def forward(self, imgs):
        distrs = self.mnist_net(imgs)
        return self.pred(digit=distrs)


def bce_loss(output, ground_truth):
    (_, dim) = output.shape
    gt = torch.stack(
        [torch.tensor([1.0 if i == t else 0.0 for i in range(dim)])
         for t in ground_truth]
    )
    # clamp to avoid saturation blow-up of BCE on exact 0/1 probabilities
    return F.binary_cross_entropy(output.clamp(1e-6, 1.0 - 1e-6), gt)


def digit_accuracy(mnist_net, root):
    """Accuracy of the underlying digit classifier on the MNIST test set."""
    dataset = torchvision.datasets.MNIST(
        str(root), train=False, transform=mnist_img_transform, download=True
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=256)
    mnist_net.eval()
    correct, count = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            pred = mnist_net(imgs).argmax(dim=-1)
            correct += (pred == labels).sum().item()
            count += len(labels)
    return correct / count


def run(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    out_dir = _HERE / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.task == "sum2":
        train_loader = torch.utils.data.DataLoader(
            MNISTSum2Dataset(str(_DATA_ROOT), train=True, transform=mnist_img_transform),
            collate_fn=MNISTSum2Dataset.collate_fn,
            batch_size=args.batch_size, shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            MNISTSum2Dataset(str(_DATA_ROOT), train=False, transform=mnist_img_transform),
            collate_fn=MNISTSum2Dataset.collate_fn,
            batch_size=256, shuffle=False,
        )
        network = MNISTSum2Net(args.provenance, args.top_k)
    else:
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(
                str(_DATA_ROOT), train=True, transform=mnist_img_transform, download=True
            ),
            batch_size=args.batch_size, shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(
                str(_DATA_ROOT), train=False, transform=mnist_img_transform, download=True
            ),
            batch_size=256, shuffle=False,
        )
        network = MNISTIdentityNet(args.provenance, args.top_k)

    optimizer = optim.Adam(network.parameters(), lr=args.learning_rate)

    t0 = time.time()
    for epoch in range(1, args.n_epochs + 1):
        network.train()
        it = tqdm(train_loader, total=len(train_loader))
        for (data, target) in it:
            optimizer.zero_grad()
            output = network(data)
            loss = bce_loss(output, target)
            loss.backward()
            optimizer.step()
            it.set_description(f"[Train Epoch {epoch}] Loss: {loss.item():.4f}")
    train_time = time.time() - t0

    # task-level accuracy
    network.eval()
    correct, count = 0, 0
    with torch.no_grad():
        for (data, target) in test_loader:
            pred = network(data).argmax(dim=-1)
            correct += (pred == target).sum().item()
            count += len(target)
    task_acc = correct / count
    dig_acc = digit_accuracy(network.mnist_net, _DATA_ROOT)

    print(f"task={args.task} seed={args.seed} task_acc={task_acc:.4f} "
          f"digit_acc={dig_acc:.4f} train_time={train_time:.1f}s")

    csv_path = out_dir / "scallop_results.csv"
    new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["task", "seed", "epochs", "provenance", "top_k",
                        "task_acc", "digit_acc", "train_time_s"])
        w.writerow([args.task, args.seed, args.n_epochs, args.provenance,
                    args.top_k, task_acc, dig_acc, round(train_time, 1)])


if __name__ == "__main__":
    parser = ArgumentParser("mnist_scallop")
    parser.add_argument("--task", choices=["digit", "sum2"], default="sum2")
    parser.add_argument("--n-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--provenance", type=str, default="difftopkproofs")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    run(args)
