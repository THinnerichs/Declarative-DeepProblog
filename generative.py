import pickle
from collections.abc import Mapping
from typing import Iterator

import torch
from problog.logic import Term, Var, Constant
from torchvision.utils import save_image

from deepproblog.engines import ExactEngine
from deepproblog.logger import VerboseLogger
from deepproblog.model import Model
from deepproblog.query import Query

import argparse
import os

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


class LatentSource(Mapping[Term, torch.Tensor]):

    def __iter__(self) -> Iterator[torch.Tensor]:
        pass

    def __init__(self, nr_embeddings=10, embedding_size=10) -> None:
        super().__init__()
        self.data = torch.nn.Embedding(nr_embeddings, embedding_size)

    def __getitem__(self, index: tuple[Term]) -> torch.Tensor:
        i = torch.LongTensor([int(index[0])])
        tensor = self.data(i)[0]
        return tensor

    def __len__(self) -> int:
        return self.data.shape

embed_size = 12

model = Model.from_file('prototype.pl', logger=VerboseLogger(1000))
if pretrain:
    latent = LatentSource(embedding_size=embed_size)
    model.add_tensor_source('prototype', latent)

    model.fit(batch_size=16, stop_condition=10)

    with open('model_prototype.dpl', 'wb') as f:
      pickle.dump(model, f)
    # dump prototype tensor source
    with open('latent_source_prototype.torch', 'wb') as f:
      pickle.dump(model.tensor_sources["prototype"], f)

with open('model_prototype.dpl', 'rb') as f:
    model2 = pickle.load(f)

model.networks = model2.networks

model.networks['encoder'].freeze()
model.networks['decoder'].freeze()
with open('latent_source_prototype.torch', 'rb') as f:
    latent = pickle.load(f)

model.add_tensor_source('prototype', latent)

from deepproblog.dataset import DataLoader, QueryDataset
from deepproblog.evaluate import get_confusion_matrix
train_set = QueryDataset(model.get_evidence())
print("Accuracy: ", get_confusion_matrix(model, train_set, verbose=1).accuracy())

raise Exception


# mnist_test = MNIST('mnist_test')

engine = ExactEngine(model)

# query = Query(Term('digit', Var('X'), Constant(7)))
# query = Query(Term('digit', Var('X'), Var('Y')))
# query = Query(Term('addition', Term('tensor', Term('mnist_train', Constant(7))), Var('Y'), Constant(8)))
query = Query(Term('addition', Var('X'), Var('Y'), Constant(7)))
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
