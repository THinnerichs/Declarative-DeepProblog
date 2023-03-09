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

# model = Model.from_file('vae.pl', logger=VerboseLogger(1000))

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
        print(tensor)
        return tensor

    def __len__(self) -> int:
        return self.data.shape


with open('model_vae.dpl', 'rb') as f:
    model = pickle.load(f)

# model.networks = model2.networks
model.freeze()
latent = LatentSource()
model.add_tensor_source('latent', latent)

optim = torch.optim.Adam(latent.data.parameters(), lr=1e-1)
# mnist_test = MNIST('mnist_test')

engine = ExactEngine(model)

query = Query(Term('digit', Var('X'), Constant(3)))
ac = engine.query(query)
for i in range(1001):
    results = ac.evaluate(model)
    for key in results:
        tensor, label = key.args
        probability = results[key]
        tensor = model.get_tensor(tensor).detach()
        loss = -torch.log(probability)
        if i % 100 == 0:
            save_image(tensor, '{}_{}.png'.format(i, key), value_range=(-1.0, 1.0))
            print(key, ':', float(probability))
            print('Loss: ', loss)
        optim.zero_grad()
        loss.backward()
        optim.step()
