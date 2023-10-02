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

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the named parameter
    save_path = args.save_path

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
        return tensor #.view(1,28,28)

    def __len__(self) -> int:
        return self.data.shape

model = Model.from_file('prototype.pl', logger=VerboseLogger(1000))
latent = LatentSource(embedding_size=10)
model.add_tensor_source('prototype', latent)

# model.fit(batch_size=16, stop_condition=10)

# with open('model_prototype.dpl', 'wb') as f:
    # pickle.dump(model, f)

with open('model_prototype.dpl', 'rb') as f:
    model2 = pickle.load(f)

model.networks = model2.networks
model.tensor_sources = model2.tensor_sources

model.networks['encoder'].freeze()
# latent = LatentSource(embedding_size=10)
# model.add_tensor_source('prototype', latent)
print(model.tensor_sources)

optim = torch.optim.Adam(latent.data.parameters(), lr=1e-4, weight_decay=1e-3)
# mnist_test = MNIST('mnist_test')

engine = ExactEngine(model)

query = Query(Term('digit', Var('X'), Constant(7)))
# query = Query(Term('addition', Term('tensor', Term('mnist_train', Constant(4))), Var('Y'), Constant(8)))
# query = Query(Term('addition', Var('X'), Var('Y'), Constant(7)))
ac = engine.query(query)
for i in range(100001):
    results = ac.evaluate(model)
    # print(results)
    key = max(results, key = lambda x: results[x])
    # for key in results:
    tensor1_term, label = key.args
    # tensor1_term, tensor2_term, label = key.args
    probability = results[key]
    tensor1 = model.get_tensor(tensor1_term).detach()
    # tensor2 = model.get_tensor(tensor2_term).detach()

    loss = -torch.log(probability)
    if i % 5000 == 0:
        save_image(tensor1, output_path + '{}_{}_term_1.png'.format(tensor1_term, i), value_range=(-1.0, 1.0))
        # save_image(tensor2, output_path + '{}_{}_term_2.png'.format(tensor2_term, i), value_range=(-1.0, 1.0))
        print(key, ':', float(probability))
        print('Loss: ', loss)
    optim.zero_grad()
    loss.backward()
    optim.step()
