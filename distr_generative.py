import pickle
from collections.abc import Mapping
from typing import Iterator

from problog.logic import Term, Var, Constant
from torchvision.utils import save_image
from json import dumps

import torch

from deepproblog.dataset import DataLoader, QueryDataset
from deepproblog.engines import ApproximateEngine, ExactEngine
from deepproblog.evaluate import get_confusion_matrix
from deepproblog.examples.MNIST.data import MNIST_train, MNIST_test, addition, MNIST
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model


import argparse
import os

method = "exact"
# problem = "digit"
problem = "addition"

if problem == "digit":
    name = f"digit_distr_{method}"
elif problem == "addition":
    N = 1
    name = f"addition_distr_{method}_{N}"
else:
    raise ValueError

save_path = ""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Learning generative DeepProbLog")

    # Add a named parameter
    parser.add_argument("--save_path", type=str, help="Path to save the output")
    parser.add_argument('--pretrain', action=argparse.BooleanOptionalAction, help='Run pre-training?', default=False)
    parser.add_argument('--show_all', action=argparse.BooleanOptionalAction, help='Write all possible groundings?', default=False)

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the named parameter
    save_path = args.save_path
    pretrain = args.pretrain
    show_all = args.show_all

epochs = 10

output_path = f"output/{save_path}"
output_path += "/" if not output_path.endswith("/") else ""

print("Output path: ", output_path)

if not os.path.exists(output_path):
    # If it doesn't exist, create it (including parent directories if needed)
    os.makedirs(output_path)

# Create LatentSource
class LatentSource(Mapping[Term, torch.Tensor]):

    def __iter__(self) -> Iterator[torch.Tensor]:
        pass

    def __init__(self, nr_embeddings=10, embedding_size=10) -> None:
        super().__init__()
        self.data = torch.nn.Embedding(nr_embeddings, embedding_size)

    def __getitem__(self, index: tuple[Term]) -> torch.Tensor:
        i = torch.LongTensor([int(index[0])])
        tensor = self.data(i)[0]
        # print(f"LatentSource:\t{i}\t{tensor}")
        return tensor

    def __len__(self) -> int:
        return self.data.shape

embed_size = 12

if problem == "digit":
    train_set = MNIST("train")
    test_set = MNIST("test")
elif problem == "addition":
    train_set = addition(N, "train")
    test_set = addition(N, "test")

from distr_prototype_networks import encoder, decoder
encoder_network, enc_opt = encoder(embed_size)
decoder_network, dec_opt = decoder(embed_size)

enc = Network(encoder_network, "encoder", batching=True)
enc.optimizer = enc_opt
dec = Network(decoder_network, "decoder", batching=True)
dec.optimizer = dec_opt

model = Model("models/distr_prototype.pl", [enc, dec])
if method == "exact":
    engine = ExactEngine(model)
    model.set_engine(engine, cache=False)
elif method == "geometric_mean":
    engine = ApproximateEngine(model, 1, ApproximateEngine.geometric_mean, exploration=False)
    model.set_engine(engine)

model.add_tensor_source("train", MNIST_train)
model.add_tensor_source("test", MNIST_test)

if pretrain:
    latent = LatentSource(embedding_size=embed_size*2) # Prototypes now have hold mean + std, hence times 2

    model.add_tensor_source('prototype', latent)
    loader = DataLoader(train_set, 2, False)
    train = train_model(model, loader, epochs, log_iter=10000, profile=0)
    model.save_state("snapshot/" + name + ".pth")
    train.logger.comment(dumps(model.get_hyperparameters()))
    train.logger.comment(
        "Accuracy {}".format(get_confusion_matrix(model, test_set, verbose=1).accuracy())
    )
    train.logger.write_to_file("log/" + name)

    # dump prototype tensor source
    with open('distr_latent_source_prototype.torch', 'wb') as f:
        pickle.dump(model.tensor_sources["prototype"], f)
else:
    with open('distr_latent_source_prototype.torch', 'rb') as f:
        latent = pickle.load(f)

    model.add_tensor_source('prototype', latent)
    model.load_state("snapshot/" + name + ".pth")

for param in latent.data.parameters():
    param.requires_grad = False
for param in model.networks['encoder'].parameters():
    param.requires_grad = False
for param in model.networks['decoder'].parameters():
    param.requires_grad = False

# model.networks['encoder'].freeze()
# model.networks['decoder'].freeze()

from deepproblog.query import Query
# query = Query(Term('digit', Var('X'), Constant(6)))

# query = Query(Term('digit', Var('X'), Var('Y')))
# dataset_name = test_set.dataset_name
# query = Query(Term('addition', Term('tensor', Term('mnist_train', Constant(7))), Var('Y'), Constant(8)))
query = Query(Term('addition', Var('X'), Var('Y'), Constant(9)))
# ac = engine.query(query)

answers = model.solve([query])[0].result

print(f"{answers=}")

groundings = answers if show_all else {max(answers, key = lambda x: answers[x]):1.0}

print(f"{groundings=}")

for key, prob in groundings.items():
    print(f"{key.args=}")
    if len(key.args) == 2:
        tensor1_term, label = key.args
        # probability = results[key]
        
        tensor1 = model.get_tensor(tensor1_term).detach()

        save_image(tensor1, output_path + '{}_term_1.png'.format(tensor1_term), value_range=(-1.0, 1.0))
    elif len(key.args) == 3:
        tensor1_term, tensor2_term, label = key.args
        
        tensor1 = model.get_tensor(tensor1_term).detach()
        tensor2 = model.get_tensor(tensor2_term).detach()

        save_image(tensor1, output_path + '{}_term_1.png'.format(tensor1_term), value_range=(-1.0, 1.0))
        save_image(tensor2, output_path + '{}_term_2.png'.format(tensor2_term), value_range=(-1.0, 1.0))
    else:
        raise ValueError("Unsupported number of arguments of result tensors.")