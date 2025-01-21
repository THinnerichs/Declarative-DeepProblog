from json import dumps
import pickle
import os.path


import torch

from deepproblog.dataset import Dataset
from deepproblog.engines import ApproximateEngine, ExactEngine
# from deepproblog.evaluate import get_confusion_matrix
#from deepproblog.examples.MNIST.data import MNIST_train, MNIST_test, addition, MNIST
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.logger import VerboseLogger

from sklearn.metrics import accuracy_score

# local imports
from data import MNIST, addition, MNIST_train, MNIST_test
from network import MNIST_Net

N = 2

name = "addition_{}_{}".format(method, N)

problem = "digit"
# problem = "addition"
if problem == "digit":
    train_set = MNIST("train")
    test_set = MNIST("test")
elif problem == "addition":
    train_set = addition(N, "train")
    test_set = addition(N, "test")

network = MNIST_Net()
pretrain = 0
if pretrain is not None and pretrain > 0:
    network.load_state_dict(torch.load("models/pretrained/all_{}.pth".format(pretrain)))
mnist_net = Network(network, "mnist_net")
mnist_net.optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

# load program
path = "models/addition.pl"
with open(path) as f:
    program_string = f.read()

# init model and engine
logger = VerboseLogger(log_every=100)
model = Model(program_string, [mnist_net], logger=logger)
engine = ExactEngine(model, cache_memory=True)


print("Training network ...")
state_file = "state_dict.pkl"
if os.path.isfile(state_file):
    with open(state_file, 'rb') as f:
        state_dict = pickle.load(f)
    model.__setstate__(state_dict)

    model.add_tensor_source("train", MNIST_train)
    model.add_tensor_source("test", MNIST_test)
else:
    model.add_tensor_source("train", MNIST_train)
    model.add_tensor_source("test", MNIST_test)

    model.fit(dataset=train_set, engine=engine, batch_size=16, shuffle=True, stop_condition=1)
    state_dict = model.__getstate__()
    with open(state_file, 'wb') as f:
        pickle.dump(state_dict, f)

# print("Making predictions")
# y_pred = model.predict(dataset=train_set, engine=engine)
# y_test = train_set.get_labels().numpy()
# accuracy = accuracy_score(y_test, y_pred)
# print("Train accuracy: \t", accuracy)

y_pred = model.predict(dataset=test_set, engine=engine)
print("predictions: ", y_pred)
y_test = test_set.get_labels().numpy()
print(y_test)

accuracy = accuracy_score(y_test, y_pred)
print("Test accuracy: \t", accuracy)