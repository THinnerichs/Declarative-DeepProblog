# run_mnistr.py
import os
import os.path
import pickle
import argparse

import torch
from sklearn.metrics import accuracy_score

from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.engines import ExactEngine
from deepproblog.logger import VerboseLogger

# your existing digit model + tensor sources
from networks.network import MNIST_Net
from data import MNIST, MNIST_train, MNIST_test

# MNIST-R datasets we created
from data import (
    MNISTNot34Binary,   # not_3_or_4(Image) -> {0,1}
    MNISTCount3s,       # count_digit_3([Imgs]) -> Count
    MNISTCount34,       # count_3_or_4([Imgs]) -> Count
    MNISTLessThanBinary # less_than(ImageA, ImageB) -> {0,1}
)

# -----------------------
# CLI
# -----------------------
parser = argparse.ArgumentParser("Run MNIST-R tasks with DeepProbLog")
parser.add_argument("--task",
                    choices=["not34", "count3", "count34", "lessthan"],
                    default="not34",
                    help="MNIST-R task to run")
parser.add_argument("--list_len", type=int, default=5,
                    help="List length for counting tasks")
parser.add_argument("--epochs", type=int, default=5,
                    help="Training stop_condition")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--state_file", type=str,
                    default="saved_models/mnistr_state.pkl")
parser.add_argument("--program", type=str,
                    default="models/mnistr.pl",
                    help="Prolog program with not_3_or_4/count/less_than rules")
args = parser.parse_args()

# -----------------------
# Pick dataset by task
# -----------------------
if args.task == "not34":
    train_set = MNISTNot34Binary("train", seed=args.seed)
    test_set  = MNISTNot34Binary("test",  seed=args.seed+1)
elif args.task == "count3":
    train_set = MNISTCount3s("train", list_len=args.list_len, seed=args.seed)
    test_set  = MNISTCount3s("test",  list_len=args.list_len, seed=args.seed+1)
elif args.task == "count34":
    train_set = MNISTCount34("train", list_len=args.list_len, seed=args.seed)
    test_set  = MNISTCount34("test",  list_len=args.list_len, seed=args.seed+1)
elif args.task == "lessthan":
    train_set = MNISTLessThanBinary("train", seed=args.seed)
    test_set  = MNISTLessThanBinary("test",  seed=args.seed+1)
else:
    raise ValueError("Unknown task")

print(f"Task: {args.task} | list_len={args.list_len} | epochs={args.epochs}")

# -----------------------
# Network & model
# -----------------------
net = MNIST_Net()
mnist_net = Network(net, "mnist_net")
mnist_net.optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

with open(args.program) as f:
    program_string = f.read()

logger = VerboseLogger(log_every=100)
model = Model(program_string, [mnist_net], logger=logger)
engine = ExactEngine(model, cache_memory=True)

# tensor sources must match tensor(Source, Id) used by your .pl
model.add_tensor_source("train", MNIST_train)
model.add_tensor_source("test", MNIST_test)

# -----------------------
# Train / Restore
# -----------------------
state_file = args.state_file
if os.path.isfile(state_file):
    with open(state_file, "rb") as f:
        state_dict = pickle.load(f)
    model.__setstate__(state_dict)
    print(f"Restored model state from {state_file}")
else:
    print("Training …")
    model.fit(dataset=train_set,
              engine=engine,
              batch_size=args.batch_size,
              shuffle=True,
              stop_condition=args.epochs)
    os.makedirs(os.path.dirname(state_file), exist_ok=True)
    with open(state_file, "wb") as f:
        pickle.dump(model.__getstate__(), f)
    print(f"Saved model state to {state_file}")

# -----------------------
# Evaluate
# -----------------------
y_pred = model.predict(dataset=test_set, engine=engine)
y_true = test_set.get_labels().numpy()

print("predictions (first 25):", y_pred[:25])
print("labels (first 25):", y_true[:25])

acc = accuracy_score(y_true, y_pred)
print(f"Test accuracy: {acc:.4f}")
