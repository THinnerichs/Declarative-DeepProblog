"""Data loaders for the DeepSeaProbLog MNIST baseline.

Adapted from examples/LOGICVAE/data/__init__.py of the official
DeepSeaProbLog repository. Reuses the MNIST copy that is already part of
this repository (deepproblog_examples/MNIST/data) with the same
normalisation to [-1, 1] as the Declarative DeepProbLog experiments.
"""
import random
from pathlib import Path

import tensorflow as tf
import torch
import torchvision
import torchvision.transforms as transforms

_DATA_ROOT = Path(__file__).parent.parent.parent / "deepproblog_examples" / "MNIST" / "data"

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

datasets = {
    "train": torchvision.datasets.MNIST(
        root=str(_DATA_ROOT), train=True, download=True, transform=transform
    ),
    "test": torchvision.datasets.MNIST(
        root=str(_DATA_ROOT), train=False, download=True, transform=transform
    ),
}


def _shuffled_indices(dataset, seed):
    indices = list(range(len(dataset)))
    if seed is not None:
        random.Random(seed).shuffle(indices)
    return indices


def digit_loader(split, seed=7, data_size=None, batch_size=10):
    """Direct supervision: batches of ([images, labels], target_probs)."""
    dataset = datasets["train" if split == "train" else "test"]
    indices = _shuffled_indices(dataset, seed)
    if data_size is not None:
        indices = indices[:data_size]

    loader = []
    for start in range(0, len(indices) - batch_size + 1, batch_size):
        batch = indices[start : start + batch_size]
        images = torch.stack([dataset[i][0] for i in batch])
        labels = torch.tensor([dataset[i][1] for i in batch])
        I = tf.reshape(tf.constant(images.numpy()), [batch_size, 28, 28, 1])
        N = tf.constant(labels.numpy(), dtype=tf.float32)
        loader.append([[I, N], tf.constant([1.0] * batch_size, dtype=tf.float32)])
    return loader


def addition_loader(split, seed=7, data_size=None, batch_size=10, curriculum=False):
    """Distant supervision: batches of ([I1, I2, Sum], target_probs).

    With curriculum=True the individual digit labels are exposed instead
    of the sum, matching the curriculum phase of the original example.
    """
    dataset = datasets["train" if split == "train" else "test"]
    indices = _shuffled_indices(dataset, seed)
    pairs = [(indices[2 * i], indices[2 * i + 1]) for i in range(len(indices) // 2)]
    if data_size is not None:
        pairs = pairs[:data_size]

    loader = []
    for start in range(0, len(pairs) - batch_size + 1, batch_size):
        batch = pairs[start : start + batch_size]
        images1 = torch.stack([dataset[i][0] for i, _ in batch])
        images2 = torch.stack([dataset[j][0] for _, j in batch])
        labels1 = torch.tensor([dataset[i][1] for i, _ in batch])
        labels2 = torch.tensor([dataset[j][1] for _, j in batch])
        I1 = tf.reshape(tf.constant(images1.numpy()), [batch_size, 28, 28, 1])
        I2 = tf.reshape(tf.constant(images2.numpy()), [batch_size, 28, 28, 1])
        d1 = tf.constant(labels1.numpy(), dtype=tf.float32)
        d2 = tf.constant(labels2.numpy(), dtype=tf.float32)
        target = tf.constant([1.0] * batch_size, dtype=tf.float32)
        if curriculum:
            loader.append([[I1, I2, d1, d2], target])
        else:
            loader.append([[I1, I2, tf.add(d1, d2)], target])
    return loader


def train_images_and_labels():
    """Full normalised training set as numpy arrays, for the 1-NN
    generative-accuracy oracle (same metric as the Declarative
    DeepProbLog experiments)."""
    dataset = datasets["train"]
    images = dataset.data.numpy().astype("float32") / 255.0
    images = (images - 0.5) / 0.5
    labels = dataset.targets.numpy()
    return images.reshape(len(images), -1), labels
