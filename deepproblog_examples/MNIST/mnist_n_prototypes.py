import pickle
from collections.abc import Mapping
from typing import Iterator


from problog.logic import Term, Var, Constant
from torchvision.utils import save_image
from json import dumps

import torch

from deepproblog.dataset import DataLoader, QueryDataset
from deepproblog.engines import ApproximateEngine, ExactEngine
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.logger import VerboseLogger

from sklearn.metrics import accuracy_score

import argparse
import os
import csv
import time

# local imports
from data import MNIST, addition, MNIST_train, MNIST_test

def load_state(model, state_file):
    with open(state_file, 'rb') as f:
        state_dict = pickle.load(f)
    model.__setstate__(state_dict)

def save_state(model, state_file):
    state_dict = model.__getstate__()
    with open(state_file, 'wb') as f:
        pickle.dump(state_dict, f)

method = "exact"

save_path = ""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Learning a declarative DeepProbLog? These are your options:")

    # Add a named parameter
    parser.add_argument("--ae_type", type=str, help="vae or ae?", default="vae")
    parser.add_argument("--model_type", type=str, help="vae or diffusion?", default="vae")
    parser.add_argument("--save_path", type=str, help="Path to save the output")
    parser.add_argument("--problem", type=str, help="digit or addition task?", default="digit")
    parser.add_argument('--inference_only', action=argparse.BooleanOptionalAction, help='Load pre-trained model and only do inference?', default=False)
    parser.add_argument('--show_all', action=argparse.BooleanOptionalAction, help='Write all possible groundings?', default=False)

    # Parse the command-line arguments
    args = parser.parse_args()

    # Access the named parameter
    save_path = args.save_path
    ae_type = args.ae_type
    model_type = args.model_type
    inference_only = args.inference_only
    show_all = args.show_all
    problem = args.problem

# Check arguments
if ae_type not in ["ae", "vae"]:
    raise ValueError("Invalid auto-encoder type selected.")
if model_type not in ["vae", "diffusion"]:
    raise ValueError("Invalid model type selected.")
if problem == "digit":
    name = f"digit_{model_type}_{method}"
elif problem == "addition":
    N = 1
    name = f"addition_{model_type}_{method}_{N}"
else:
    raise ValueError

output_path = f"output/{save_path}"
output_path += "/" if not output_path.endswith("/") else ""

print("Output path: ", output_path)

# If it doesn't exist, create it (including parent directories if needed)
if not os.path.exists(output_path):
    os.makedirs(output_path)

class NPrototypesLatentSource(Mapping):
    """
    Tensor source for terms of the form tensor(prototype(C,K)).
    Holds nr_digits * nr_prototypes learnable embeddings, optionally sized for VAE (mean+std).
    """

    def __init__(self, nr_digits=10, nr_prototypes=3, embedding_size=10) -> None:
        super().__init__()
        self.nr_digits = nr_digits
        self.nr_prototypes = nr_prototypes
        self.data = torch.nn.Embedding(nr_digits * nr_prototypes, embedding_size)

    # --- helpers ---
    @staticmethod
    def _to_int(x):
        # Works for ProbLog Constant/Term or plain ints/strings
        try:
            return int(x)
        except Exception:
            return int(str(x))

    def _flatten_index(self, index):
        """
        Accepts:
          - tuple like (C,K)
          - single value C (backward-compat: maps to K=0)
        Returns the flattened id: C * nr_prototypes + K
        """
        if isinstance(index, tuple):
            args = tuple(self._to_int(a) for a in index)
        else:
            args = (self._to_int(index),)

        if len(args) == 2:
            c, k = args
        elif len(args) == 1:
            c, k = args[0], 0  # fallback for old unary prototype
        else:
            raise ValueError(f"Expected 1 or 2 indices, got {len(args)}: {args}")

        if not (0 <= c < self.nr_digits):
            raise IndexError(f"Class index out of range: {c}")
        if not (0 <= k < self.nr_prototypes):
            raise IndexError(f"Prototype index out of range: {k}")

        return c * self.nr_prototypes + k

    # --- Mapping interface ---
    def __getitem__(self, index):
        flat = self._flatten_index(index)
        i = torch.LongTensor([flat])
        return self.data(i)[0]

    def __len__(self) -> int:
        return self.data.num_embeddings

    def __iter__(self):
        # Not used by DeepProbLog; included for completeness
        return iter(range(self.__len__()))

embed_size = 12
n_prototypes = 3

if problem == "digit":
    train_set = MNIST("train")
    test_set = MNIST("test")
elif problem == "addition":
    train_set = addition(N, "train")
    test_set = addition(N, "test")

if model_type == "vae":
    from networks.VAE_networks import encoder, decoder
    encoder_network, enc_opt = encoder(embed_size)
    decoder_network, dec_opt = decoder(embed_size)
elif model_type == "diffusion":
    from networks.DDPM_networks import encoder, decoder 
    encoder_network, enc_opt = encoder(embed_size)
    decoder_network, dec_opt = decoder(embed_size)

enc = Network(encoder_network, "encoder")
enc.optimizer = enc_opt
dec = Network(decoder_network, "decoder")
dec.optimizer = dec_opt

# load program
path = f"models/n_prototype_{ae_type}.pl" # for n_prototypes
with open(path) as f:
    program_string = f.read()

logger = VerboseLogger(log_every=100)
model_path = f"saved_models/{problem}_{model_type}_model_save_dict.pkl"

latent_file = f'saved_models/{model_type}_latent_source_prototype_3x.torch'


if inference_only:
    model = Model(program_string, [enc, dec], logger=logger)
    engine = ExactEngine(model, cache_memory=True)

    # Load pretrained model
    with open(latent_file, 'rb') as f:
        latent = pickle.load(f)
    load_state(model, model_path)
    model.add_tensor_source('prototype', latent)

    model.add_tensor_source("train", MNIST_train)
    model.add_tensor_source("test", MNIST_test)

else:
    model = Model(program_string, [enc, dec], logger=logger)
    engine = ExactEngine(model, cache_memory=True)

    model.add_tensor_source("train", MNIST_train)
    model.add_tensor_source("test", MNIST_test)
    
    # Run training
    latent = NPrototypesLatentSource(nr_digits=10, nr_prototypes=3, embedding_size=embed_size*2)

    num_epochs = 5 
    model.add_tensor_source('prototype', latent)
    model.fit(dataset=train_set, engine=engine, batch_size=16, shuffle=True, stop_condition=num_epochs)

    # prototype tensor source
    with open(latent_file, 'wb') as f:
        pickle.dump(model.tensor_sources["prototype"], f)
    save_state(model, model_path)

    y_pred = model.predict(dataset=test_set, engine=engine)
    y_test = test_set.get_labels().numpy()

    accuracy = accuracy_score(y_test, y_pred)
    print("Test accuracy: \t", accuracy)

    # Get accuracy to put for RQ1
    filename = f'n_prototypes_{name}_RQ1.csv'

    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Append the data
        writer.writerow([accuracy])


# Run inference
for param in latent.data.parameters():
    param.requires_grad = False
for param in model.networks['encoder'].parameters():
    param.requires_grad = False
for param in model.networks['decoder'].parameters():
    param.requires_grad = False

from deepproblog.query import Query

# For RQ3.2:
run_RQ3_2 = False
if run_RQ3_2:
    # Setup
    import random
    n = 100
    number_length = 4
    values_to_mask = 4
    dataset = addition(number_length, "test", seed=42)
    labels = train_set.get_labels()

    # Computation
    correct_queries = 0
    for i in range(n):
        # Get query from dataset
        query = dataset.to_query(random.randint(1, len(dataset)))
        sub_dict = query.substitution

        # Mask `values_to_mask` elements
        keys_to_mask = random.sample(list(sub_dict.keys()), values_to_mask)

        masked_values = {key: sub_dict[key] for key in keys_to_mask}

        masked_sub_dict = {
            key: (Var(f"{key.functor.upper()}") if key in keys_to_mask else value)
            for key, value in sub_dict.items()
        }
        query.substitution = masked_sub_dict

        # Generate images
        start_time = time.time()
        answers = model.query(query, engine).result

        groundings = {max(answers, key = lambda x: answers[x]):1.0}
        
        # Get labels of closest images
        correct_preds = 0
        for orig_key, grounding_key in zip(keys_to_mask, groundings):
            tensor1_term, label = grounding_key.args
            tensor1 = model.get_tensor(tensor1_term).detach()

            best_im, best_y = None, None
            best_distance = float('inf')
            for im, y in train_set.data:
                # Calculate Euclidean distance
                distance = torch.norm(tensor1 - im)
                
                # Check if this image is closer than the ones checked before
                if distance < best_distance:
                    best_distance = distance
                    best_im = im
                    best_y = y

            # Add truth to list
            if best_y == label[masked_values[orig_key].value]:
                correct_preds += 1
        
        if correct_preds == values_to_mask:
            correct_queries += 1
    # Compute accuracy
    accuracy = correct_queries / n
    filename = f'{name}_RQ3_2.csv'

    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(accuracy)


# Here are some sample queries. Un-comment a query for it to be answered.
# query = Query(Term('digit', Var('X'), Constant(6)))
# query = Query(Term('digit', Var('X'), Var('Y')))
# query = Query(Term('addition', Term('tensor', Term('train', Constant(7))), Var('Y'), Constant(8)))
query = Query(Term('addition', Var('X'), Var('Y'), Constant(9)))
# query = Query(Term('addition', Var('X'), Var('Y'), Var('Z')))

answers = model.query(query, engine).result

print(f"{answers=}")

if show_all:
    groundings = {k:v for k,v in answers.items()}
else:
    groundings = {max(answers, key = lambda x: answers[x]):1.0}

run_RQ3_1 = True

for key, prob in groundings.items():
    print(f"{key.args=}")
    if len(key.args) == 2:
        # Digit case
        tensor1_term, label = key.args
        # probability = results[key]
        
        tensor1 = model.get_tensor(tensor1_term).detach()

        if run_RQ3_1:
            best_im, best_y = None, None
            start_time = time.time()

            best_distance = float('inf')
        
            for im, y in train_set.data:
                # Calculate Euclidean distance
                distance = torch.norm(tensor1 - im)
                
                # Check if this image is closer than the ones checked before
                if distance < best_distance:
                    best_distance = distance
                    best_im = im
                    best_y = y

            print(f"This took {time.time() - start_time} seconds.")
            print("Label:", label, "closest y:", best_y)
            filename = f'{name}_RQ3_1.csv'

            # Open the file in append mode
            with open(filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([label, best_y])

        image_path = output_path + '{}_term_1.png'.format(tensor1_term)
        save_image(tensor1, image_path, value_range=(-1.0, 1.0))
        print("Saved image to", image_path)
    elif len(key.args) == 3:
        # Addition case
        tensor1_term, tensor2_term, label = key.args
        
        tensor1 = model.get_tensor(tensor1_term).detach()
        tensor2 = model.get_tensor(tensor2_term).detach()

        if run_RQ3_1:
            predicted_labels = []
            for i, tensor in enumerate([tensor1, tensor2]):
                best_im, best_y = None, None
                start_time = time.time()

                best_distance = float('inf')
            
                for im, y in train_set.dataset:
                    # Calculate Euclidean distance
                    distance = torch.norm(tensor - im)
                    
                    # Check if this image is closer than the ones checked before
                    if distance < best_distance:
                        best_distance = distance
                        best_im = im
                        best_y = y

                predicted_labels.append(best_y)

            print("Sum:", label, "predicted sum:", sum(predicted_labels))
            filename = f'{name}_RQ3_1.csv'

            # Open the file in append mode
            with open(filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([label, sum(predicted_labels)])

        save_image(tensor1, output_path + '{}_term_1.png'.format(tensor1_term), value_range=(-1.0, 1.0))
        save_image(tensor2, output_path + '{}_term_2.png'.format(tensor2_term), value_range=(-1.0, 1.0))
    else:
        raise ValueError("Unsupported number of arguments of result tensors.")


