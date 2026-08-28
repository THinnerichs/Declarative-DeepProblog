"""DeepSeaProbLog baseline on MNIST digit/2 (direct supervision) and
add/3 (distant supervision), with discriminative and generative
evaluation.

The program, architectures and training recipe follow the official
LOGICVAE example of the DeepSeaProbLog repository (De Smet et al.,
UAI 2023), with subtraction replaced by addition. Generative accuracy is
computed with the same 1-nearest-neighbour oracle as the Declarative
DeepProbLog experiments: a generated image counts as correct if its
nearest training image (L2 in pixel space, [-1, 1] normalisation) has
the queried label; an add/3 generation is correct only if the two
nearest-neighbour labels sum to the queried value.

Usage:
    python run_mnist_dsp.py --task digit    --epochs 2 --data-size 30000
    python run_mnist_dsp.py --task addition --epochs 2 --data-size 15000
"""
import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent / "deepseaproblog"))
sys.path.insert(0, str(_HERE))

import tensorflow as tf
from problog.logic import Constant, Var

from architectures import DigitClassifier, DenseEncoder, DenseDecoder
from data import addition_loader, digit_loader, train_images_and_labels
from engines.tensor_ops import Equals, SoftUnification
from model import Model
from network import Network

SHAPE_LATENT_DIM = 4
LOGIC_LATENT_DIM = 4
SAMPLE_SIZE = 50


def build_model(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    classifier = DigitClassifier(latent_dim=LOGIC_LATENT_DIM)
    encoder = DenseEncoder(latent_dim=SHAPE_LATENT_DIM)
    decoder = DenseDecoder()

    nets = [
        Network(encoder, "encoder_net"),
        Network(decoder, "decoder_net"),
        Network(classifier, "mnist_class"),
        Network(Equals(maxmult=0.2), "equals"),
        Network(SoftUnification(maxmult=4.0, n=1), "unification"),
    ]
    model = Model(
        str(_HERE / "models" / "mnist_generation.pl"), nets, nb_samples=SAMPLE_SIZE
    )
    model.set_loss(tf.keras.losses.BinaryCrossentropy(from_logits=False))
    model.set_optimizer(tf.keras.optimizers.legacy.Adam(learning_rate=1e-3))
    return model, classifier, encoder, decoder


def classifier_digit_accuracy(classifier, batch_size=100):
    from data import datasets
    import torch

    dataset = datasets["test"]
    correct, count = 0, 0
    images = dataset.data.numpy().astype("float32") / 255.0
    images = (images - 0.5) / 0.5
    labels = dataset.targets.numpy()
    for start in range(0, len(images), batch_size):
        I = tf.constant(images[start : start + batch_size, :, :, None])
        pred = tf.argmax(classifier.call(I), axis=-1).numpy()
        correct += (pred == labels[start : start + batch_size]).sum()
        count += len(pred)
    return correct / count


def classifier_addition_accuracy(classifier, seed, batch_size=100):
    """Most-probable-sum accuracy on test pairs; with a categorical digit
    distribution this equals argmax over sum probabilities of the
    image_addition query."""
    from data import datasets

    dataset = datasets["test"]
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    images = dataset.data.numpy().astype("float32") / 255.0
    images = (images - 0.5) / 0.5
    labels = dataset.targets.numpy()

    pairs = [(indices[2 * i], indices[2 * i + 1]) for i in range(len(indices) // 2)]
    correct = 0
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start : start + batch_size]
        I1 = tf.constant(np.stack([images[i] for i, _ in chunk])[:, :, :, None])
        I2 = tf.constant(np.stack([images[j] for _, j in chunk])[:, :, :, None])
        p1 = classifier.call(I1).numpy()
        p2 = classifier.call(I2).numpy()
        # distribution over sums via convolution of the two digit distributions
        sum_probs = np.zeros((len(chunk), 19))
        for a in range(10):
            for b in range(10):
                sum_probs[:, a + b] += p1[:, a] * p2[:, b]
        pred = sum_probs.argmax(axis=-1)
        true = np.array([labels[i] + labels[j] for i, j in chunk])
        correct += (pred == true).sum()
    return correct / len(pairs)


class NNOracle:
    def __init__(self):
        self.X, self.y = train_images_and_labels()  # (60000, 784) in [-1, 1]
        self.sq_norms = (self.X ** 2).sum(axis=1)

    def labels(self, generated):
        """generated: (n, 784) in [-1, 1] -> nearest-neighbour labels."""
        d = self.sq_norms[None, :] - 2.0 * generated @ self.X.T
        return self.y[d.argmin(axis=1)]


def _extract_generations(model, result, arg_indices):
    """Return, per grounding, a list of (n_samples, 784) arrays for each
    requested query-argument index."""
    out = []
    for grounding in result[0].result.keys():
        tensors = []
        for idx in arg_indices:
            t = model.get_tensor(grounding.args[idx]).numpy()
            tensors.append(t.reshape(-1, 784))
        out.append(tensors)
    return out


def generative_digit_accuracy(model, oracle, out_dir, repeats=100):
    rows, correct, total = [], 0, 0
    for d in range(10):
        for _ in range(repeats):
            result = model.solve_query(
                "generate_digit", [Constant(d), Var("X")], generate=True
            )
            for tensors in _extract_generations(model, result, [-1]):
                nn = oracle.labels(tensors[0])
                correct += (nn == d).sum()
                total += len(nn)
                rows += [[d, int(l)] for l in nn]
    with open(out_dir / "generative_digit_details.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "nn_label"])
        w.writerows(rows)
    return correct / total


def generative_addition_accuracy(model, oracle, out_dir, repeats=10):
    rows, correct, total = [], 0, 0
    for s in range(19):
        for _ in range(repeats):
            result = model.solve_query(
                "generate_addition", [Constant(s), Var("X1"), Var("X2")],
                generate=True,
            )
            for tensors in _extract_generations(model, result, [-2, -1]):
                nn1 = oracle.labels(tensors[0])
                nn2 = oracle.labels(tensors[1])
                n = min(len(nn1), len(nn2))
                correct += (nn1[:n] + nn2[:n] == s).sum()
                total += n
                rows += [[s, int(a), int(b)] for a, b in zip(nn1[:n], nn2[:n])]
    with open(out_dir / "generative_addition_details.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sum", "nn_label_1", "nn_label_2"])
        w.writerows(rows)
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["digit", "addition"], default="digit")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--data-size", type=int, default=None,
                        help="number of training examples (images or pairs)")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--curriculum", type=int, default=0,
                        help="number of labelled pairs for the curriculum "
                             "phase of the addition task (0 = none; the "
                             "original example uses 256)")
    parser.add_argument("--log-its", type=int, default=100)
    args = parser.parse_args()

    out_dir = _HERE / "results" / f"{args.task}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    model, classifier, encoder, decoder = build_model(args.seed)

    t0 = time.time()
    if args.task == "digit":
        data = digit_loader("train", seed=args.seed, data_size=args.data_size,
                            batch_size=args.batch_size)
        inputs = [tf.keras.Input([28, 28, 1]), tf.keras.Input([1])]
        model.compile_query("encode_decode_digit", inputs)
        model.train(data, args.epochs, log_its=args.log_its)
    else:
        if args.curriculum > 0:
            cur = addition_loader("train", seed=args.seed,
                                  data_size=args.curriculum, batch_size=4,
                                  curriculum=True)
            cur_inputs = [tf.keras.Input([28, 28, 1]), tf.keras.Input([28, 28, 1]),
                          tf.keras.Input([1]), tf.keras.Input([1])]
            model.compile_query("image_addition_curriculum", cur_inputs)
            model.train(cur, 1, log_its=args.log_its)
            model.set_optimizer(tf.keras.optimizers.legacy.Adam(learning_rate=1e-3))
        data = addition_loader("train", seed=args.seed, data_size=args.data_size,
                               batch_size=args.batch_size)
        inputs = [tf.keras.Input([28, 28, 1]), tf.keras.Input([28, 28, 1]),
                  tf.keras.Input([1])]
        model.compile_query("encode_decode_addition", inputs)
        model.train(data, args.epochs, log_its=args.log_its)
    train_time = time.time() - t0
    print(f"Training took {train_time:.1f}s")

    weight_dir = _HERE / "saved_networks" / f"{args.task}_seed{args.seed}"
    weight_dir.mkdir(parents=True, exist_ok=True)
    classifier.save_weights(str(weight_dir / "classifier"))
    encoder.save_weights(str(weight_dir / "encoder"))
    decoder.save_weights(str(weight_dir / "decoder"))

    digit_acc = classifier_digit_accuracy(classifier)
    add_acc = classifier_addition_accuracy(classifier, args.seed)
    print(f"Discriminative digit accuracy: {digit_acc:.4f}")
    print(f"Discriminative addition accuracy: {add_acc:.4f}")

    oracle = NNOracle()
    gen_digit = generative_digit_accuracy(model, oracle, out_dir)
    print(f"Generative digit accuracy: {gen_digit:.4f}")
    gen_add = generative_addition_accuracy(model, oracle, out_dir)
    print(f"Generative addition accuracy: {gen_add:.4f}")

    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "seed", "epochs", "data_size", "curriculum",
                    "train_time_s", "class_digit_acc", "class_add_acc",
                    "gen_digit_acc", "gen_add_acc"])
        w.writerow([args.task, args.seed, args.epochs, args.data_size,
                    args.curriculum, round(train_time, 1), digit_acc, add_acc,
                    gen_digit, gen_add])
    print("Wrote", out_dir / "summary.csv")


if __name__ == "__main__":
    main()
