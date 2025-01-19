from json import dumps

import torch

from deepproblog.dataset import Dataset
from deepproblog.engines import ApproximateEngine, ExactEngine
# from deepproblog.evaluate import get_confusion_matrix
#from deepproblog.examples.MNIST.data import MNIST_train, MNIST_test, addition, MNIST
from deepproblog.model import Model
from deepproblog.network import Network

# local imports
from data import MNIST, addition, MNIST_train, MNIST_test
from network import MNIST_Net

method = "exact"
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
# mnist_net.optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

# load program
path = "models/addition.pl"
with open(path) as f:
    program_string = f.read()

# init model and engine
model = Model(program_string, [mnist_net])
engine = ExactEngine(model, cache_memory=True)

model.add_tensor_source("train", MNIST_train)
model.add_tensor_source("test", MNIST_test)

model.fit(dataset=train_set, engine=engine, batch_size=16, shuffle=False)

#model.save_state("snapshot/" + name + ".pth")
# train.logger.comment(dumps(model.get_hyperparameters()))
# train.logger.comment(
#     "Accuracy {}".format(get_confusion_matrix(model, test_set, verbose=1).accuracy())
# )
# train.logger.write_to_file("log/" + name)
