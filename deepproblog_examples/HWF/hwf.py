from json import dumps
from torch.optim import Adam
from typing import Iterator

import pickle
from collections.abc import Mapping
from problog.logic import Term, Var, Constant
from torchvision.utils import save_image
from json import dumps

import torch

from deepproblog.dataset import DataLoader
from deepproblog.engines import ApproximateEngine, ExactEngine
from deepproblog.evaluate import get_confusion_matrix
from data import HWFDataset, hwf_images
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.train import train_model

import argparse
import os

N = 1
method = 'exact'
name = "hwf_{}_{}".format(method, N)
curriculum = False

prot_types = ["num", "op"]

print("Training HWF with N={} and curriculum={}".format(N, curriculum))

save_path = ""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Learning generative DeepProbLog")

    # Add a named parameter
    parser.add_argument("--save_path", type=str, help="Path to save the output")
    parser.add_argument("--model_type", type=str, help="ae or vae?")
    parser.add_argument('--inference_only', action=argparse.BooleanOptionalAction, help='Load pre-trained model and only do inference?', default=False)
    parser.add_argument('--load_pretrained', action=argparse.BooleanOptionalAction, help='Load existing NNs and continue training?', default=False)
    parser.add_argument('--show_all', action=argparse.BooleanOptionalAction, help='Write all possible groundings?', default=False)

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the named parameter
    save_path = args.save_path
    model_type = args.model_type
    inference_only = args.inference_only
    load_pretrained = args.load_pretrained
    show_all = args.show_all

# if model_type not in ["ae", "vae"]:
    # raise ValueError("Invalid model_type selected.")

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

from network import encoder, decoder
encoder_network, enc_opt = encoder(embed_size)
decoder_network, dec_opt = decoder(embed_size)

enc = Network(encoder_network, "encoder", batching=True)
enc.optimizer = enc_opt
dec = Network(decoder_network, "decoder", batching=True)
dec.optimizer = dec_opt

model = Model("model.pl", [enc, dec])

if method == "exact":
    model.set_engine(ExactEngine(model), cache=True)
elif method == "approximate":
    heuristic = ApproximateEngine.geometric_mean
    model.set_engine(ApproximateEngine(model, 1, heuristic, timeout=30, ignore_timeout=True, exploration=True))
model.add_tensor_source("hwf", hwf_images)
model.add_tensor_source("hwf", hwf_images)
print(model.tensor_sources["hwf"].data)

try:
    if curriculum:
        dataset = HWFDataset("train2", lambda x: x <= N)
        val_dataset = HWFDataset("val", lambda x: x <= N)
        test_dataset = HWFDataset("test", lambda x: x <= N)
    else:
        dataset = HWFDataset("train2", lambda x: x == N)
        val_dataset = HWFDataset("val", lambda x: x == N)
        test_dataset = HWFDataset("test", lambda x: x == N)
except FileNotFoundError:
    print('The HWD dataset has not been downloaded. See the README.md for info on how to download it.')
    dataset, val_dataset, test_dataset = None, None, None
    exit(1)


if inference_only:
    for prot_type in prot_types:
        with open(f'{model_type}_{prot_type}_prototypes.torch', 'rb') as f:
            latent = pickle.load(f)
        model.add_tensor_source(f'{prot_type}_prototype', latent)

    model.load_state("snapshot/" + name + ".pth")
else:
    if load_pretrained:
        for prot_type in prot_types:
            with open(f'{model_type}_{prot_type}_prototypes.torch', 'rb') as f:
                latent = pickle.load(f)
            model.add_tensor_source(f'{prot_type}_prototype', latent)
            
        model.load_state("snapshot/" + name.replace("vae", "ae") + ".pth")
    else:
        for prot_type in prot_types:
            latent = LatentSource(embedding_size=embed_size*2) # Prototypes now have hold mean + std, hence times 2
            model.add_tensor_source(f'{prot_type}_prototype', latent)

    loader = DataLoader(dataset, 32, shuffle=True)

    print("Training on size {}".format(N))
    train_log = train_model(
        model,
        loader,
        2,
        log_iter=50,
        inital_test=False
    )
    print(model.tensor_sources["hwf"].data)
    val_acc = get_confusion_matrix(x, val_dataset, eps=1e-6).accuracy()
    test_acc = get_confusion_matrix(x, test_dataset, eps=1e-6).accuracy()
    print(f"Validation acc: {val_acc}")

    model.save_state("snapshot/" + name + ".pth")

    # dump prototype tensor source
    for prot_type in prot_types:
        with open(f'{model_types}_{prot_type}_prototypes.torch', 'wb') as f:
            pickle.dump(model.tensor_sources[f"{prot_type}_prototype"], f)



model.save_state("models/" + name + ".pth")
final_acc = get_confusion_matrix(model, test_dataset, eps=1e-6, verbose=0).accuracy()
train_log.logger.comment("Accuracy {}".format(final_acc))
train_log.logger.comment(dumps(model.get_hyperparameters()))
train_log.write_to_file("log/" + name)

# Run inference
for param in latent.data.parameters():
    param.requires_grad = False
for param in model.networks['encoder'].parameters():
    param.requires_grad = False
for param in model.networks['decoder'].parameters():
    param.requires_grad = False

from deepproblog.query import Query

# query = Query(Term('expression', Var('X'), Var('Y')))
query = Query(Term('expression', Var('X'), Constant(3)))
# dataset_name = test_set.dataset_name
# query = Query(Term('addition', Term('tensor', Term('mnist_train', Constant(7))), Var('Y'), Constant(8)))
# query = Query(Term('addition', Var('X'), Var('Y'), Constant(9)))
# ac = engine.query(query)

answers = model.solve([query])[0].result

print(f"{answers=}")

groundings = answers if show_all else {max(answers, key = lambda x: answers[x]):1.0}

# Get accuracy to put for RQ1
accuracy = get_confusion_matrix(model, test_set, verbose=1).accuracy()
filename = f'{name}_RQ1.csv'

with open(filename, mode='a', newline='') as file:
    writer = csv.writer(file)
    # Append the data
    writer.writerow([accuracy])


for key, prob in groundings.items():
    print(f"{key.args=}")
    if len(key.args) == 2:
        tensor1_term, label = key.args
        # probability = results[key]
        
        tensor1 = model.get_tensor(tensor1_term).detach()


        best_im, best_y = None, None
        if problem == "digit":
            best_distance = float('inf')

            for im, y in train_set.data:
                # Calculate Euclidean distance
                distance = torch.norm(tensor1 - im)
                
                # Check if this image is closer than the ones checked before
                if distance < best_distance:
                    best_distance = distance
                    best_im = im
                    best_y = y

        print("Label:", label, "closest y:", best_y)
        filename = f'{name}_RQ2.csv'

        # Open the file in append mode
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([label, best_y])

        save_image(tensor1, output_path + '{}_term_1.png'.format(tensor1_term), value_range=(-1.0, 1.0))
    elif len(key.args) == 3:
        tensor1_term, tensor2_term, label = key.args
        
        tensor1 = model.get_tensor(tensor1_term).detach()
        tensor2 = model.get_tensor(tensor2_term).detach()

        save_image(tensor1, output_path + '{}_term_1.png'.format(tensor1_term), value_range=(-1.0, 1.0))
        save_image(tensor2, output_path + '{}_term_2.png'.format(tensor2_term), value_range=(-1.0, 1.0))
    else:
        raise ValueError("Unsupported number of arguments of result tensors.")

