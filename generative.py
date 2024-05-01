import pickle
from collections.abc import Mapping
from typing import Iterator

from problog.logic import Term, Var, Constant
from torchvision.utils import save_image
from json import dumps

import torch

from deepproblog.dataset import DataLoader
from deepproblog.engines import ApproximateEngine, ExactEngine
from deepproblog.evaluate import get_confusion_matrix
from deepproblog.examples.MNIST.data import MNIST_train, MNIST_test, addition, MNIST
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model


import argparse
import os

method = "exact"
N = 1

name = "addition_{}_{}".format(method, N)

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

train_set = addition(N, "train")
test_set = addition(N, "test")
# train_set = MNIST("train")
# test_set = MNIST("test")

from prototype_networks import encoder, decoder
encoder_network, enc_opt = encoder(embed_size)
decoder_network, dec_opt = decoder(embed_size)

enc = Network(encoder_network, "encoder", batching=True)
enc.optimizer = enc_opt
dec = Network(decoder_network, "decoder", batching=True)
dec.optimizer = dec_opt

model = Model("models/prototype.pl", [enc, dec])
if method == "exact":
    engine = ExactEngine(model)
    model.set_engine(engine, cache=False)
elif method == "geometric_mean":
    engine = ApproximateEngine(model, 1, ApproximateEngine.geometric_mean, exploration=False)
    model.set_engine(engine)

model.add_tensor_source("train", MNIST_train)
model.add_tensor_source("test", MNIST_test)

if pretrain:
    epochs = 2
    # network.load_state_dict(torch.load("models/pretrained/all_{}.pth".format(pretrain)))
    latent = LatentSource(embedding_size=embed_size)
    model.add_tensor_source('prototype', latent)

    loader = DataLoader(train_set, 2, False)
    train = train_model(model, loader, epochs, log_iter=200, profile=0)
    model.save_state("snapshot/" + name + ".pth")
    train.logger.comment(dumps(model.get_hyperparameters()))
    train.logger.comment(
        "Accuracy {}".format(get_confusion_matrix(model, test_set, verbose=1).accuracy())
    )
    train.logger.write_to_file("log/" + name)

    with open('model_prototype.dpl', 'wb') as f:
        pickle.dump(model, f)
    # dump prototype tensor source
    with open('latent_source_prototype.torch', 'wb') as f:
        pickle.dump(model.tensor_sources["prototype"], f)
else:
    with open('latent_source_prototype.torch', 'rb') as f:
        latent = pickle.load(f)

raise Exception

with open('model_prototype.dpl', 'rb') as f:
    model2 = pickle.load(f)

model.networks = model2.networks
model.add_tensor_source('prototype', latent)
model.networks['encoder'].freeze()
model.networks['decoder'].freeze()

train_set = QueryDataset(model.get_evidence())

from deepproblog.query import Query
query = Query(Term('digit', Var('X'), Constant(7)))
# query = Query(Term('digit', Var('X'), Var('Y')))
# query = Query(Term('addition', Term('tensor', Term('mnist_train', Constant(7))), Var('Y'), Constant(8)))
# query = Query(Term('addition', Var('X'), Var('Y'), Constant(7)))
ac = engine.query(query)

results = ac.evaluate(model)
groundings = results if show_all else [max(results, key = lambda x: results[x])]

for key in groundings:
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
