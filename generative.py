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

model = Model.from_file('vae.pl', logger=VerboseLogger(1000))

#
# model.fit(batch_size=4, stop_condition=1)
# with open('model_vae.dpl', 'wb') as f:
#     pickle.dump(model, f)


class LatentSource(Mapping[Term, torch.Tensor]):

    def __iter__(self) -> Iterator[torch.Tensor]:
        pass

    def __init__(self, nr_embeddings=10, embedding_size=2) -> None:
        super().__init__()
        self.data = torch.nn.Embedding(nr_embeddings, embedding_size)

    def __getitem__(self, index: tuple[Term]) -> torch.Tensor:
        i = torch.LongTensor([int(index[0])])
        tensor = self.data(i)[0]
        return tensor

    def __len__(self) -> int:
        return self.data.shape


with open('model_vae.dpl', 'rb') as f:
    model2 = pickle.load(f)

model.networks = model2.networks
model.freeze()
latent = LatentSource()
model.add_tensor_source('latent', latent)

optim = torch.optim.Adam(latent.data.parameters(), lr=1e-2)
# mnist_test = MNIST('mnist_test')

engine = ExactEngine(model)

query = Query(Term('addition', Var('X'), Var('Y'), Constant(8)))
ac = engine.query(query)
for i in range(1001):
    results = ac.evaluate(model)
    key = max(results, key = lambda x: results[x])
    # for key in results:
    tensor1_term, tensor2_term, label = key.args
    probability = results[key]
    tensor1 = model.get_tensor(tensor1_term).detach()
    tensor2 = model.get_tensor(tensor2_term).detach()

    loss = -torch.log(probability)
    if i % 200 == 0:
        save_image(tensor1, '{}_{}.png'.format(tensor1_term, i), value_range=(-1.0, 1.0))
        save_image(tensor2, '{}_{}.png'.format(tensor2_term, i), value_range=(-1.0, 1.0))
        print(key, ':', float(probability))
        print('Loss: ', loss)
    optim.zero_grad()
    loss.backward()
    optim.step()
