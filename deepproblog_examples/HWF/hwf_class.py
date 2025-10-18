# run_hwf.py
from json import dumps
import pickle
import os
import os.path

import torch
from torch.optim import Adam

from deepproblog.dataset import Dataset
from deepproblog.engines import ApproximateEngine, ExactEngine
# from deepproblog.evaluate import get_confusion_matrix
#from deepproblog.examples.MNIST.data import MNIST_train, MNIST_test, addition, MNIST
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.logger import VerboseLogger

from sklearn.metrics import accuracy_score

# HWF bits
from data import HWFDataset, hwf_images
from networks.network import SymbolEncoder, SymbolClassifier


# -----------------------
# Config
# -----------------------
N = 1                 # figure size
curriculum = False    # True -> x <= N ; False -> x == N
method = "exact"      # "exact" or "approximate"

# Paths
program_path = "models/vanilla_model.pl"  # your HWF Prolog program file
state_file = f"saved_models/hwf_{method}_N{N}_state.pkl"

# -----------------------
# Data
# -----------------------
try:
    if curriculum:
        train_set = HWFDataset("train2", lambda x: x <= N)
        val_set   = HWFDataset("val",    lambda x: x <= N)
        test_set  = HWFDataset("test",   lambda x: x <= N)
    else:
        train_set = HWFDataset("train2", lambda x: x == N)
        val_set   = HWFDataset("val",    lambda x: x == N)
        test_set  = HWFDataset("test",   lambda x: x == N)
except FileNotFoundError:
    raise SystemExit("The HWF dataset has not been downloaded. See the README.md for info on how to download it.")

# Networks
encoder = SymbolEncoder()
net_digits = SymbolClassifier(encoder, 10)
net_ops    = SymbolClassifier(encoder, 4)

net1 = Network(net_digits, "net1")
net2 = Network(net_ops, "net2")

net1.optimizer = torch.optim.Adam(net_digits.parameters(), lr=3e-3)
net2.optimizer = torch.optim.Adam(net_ops.parameters(),    lr=3e-3)

# Program & Model
with open(program_path, "r") as f:
    program_string = f.read()

logger = VerboseLogger(log_every=100)
model = Model(program_string, [net1, net2], logger=logger)

# Engine
if method == "exact":
    engine = ExactEngine(model, cache_memory=True)
else:
    engine = ApproximateEngine(model, 1, ApproximateEngine.geometric_mean,
                               timeout=30, ignore_timeout=True, exploration=True)

# Train on 10000 samples
print(f"Training HWF with N={N} and curriculum={curriculum} using {method} engine")

if os.path.isfile(state_file):
    # restore everything (model + networks + parameters + cache)
    with open(state_file, "rb") as f:
        state_dict = pickle.load(f)
    model.__setstate__(state_dict)

    model.add_tensor_source("hwf", hwf_images)
else:
    model.add_tensor_source("hwf", hwf_images)

    model.fit(dataset=train_set, engine=engine, batch_size=32, shuffle=True, stop_condition=25)
    # persist full model state (more robust than only .pth weights)
    state_dict = model.__getstate__()
    os.makedirs(os.path.dirname(state_file), exist_ok=True)
    with open(state_file, "wb") as f:
        pickle.dump(state_dict, f)

# Predict on validation 
y_val_pred = model.predict(dataset=val_set, engine=engine)
y_val_true = val_set.get_labels().numpy()
val_acc = accuracy_score(y_val_true, y_val_pred)
print("Val accuracy:\t", val_acc)

# Predict on test
# 2000 samples
y_test_pred = model.predict(dataset=test_set, engine=engine)
y_test_true = test_set.get_labels().numpy()
test_acc = accuracy_score(y_test_true, y_test_pred)
print("Test accuracy:\t", test_acc)

