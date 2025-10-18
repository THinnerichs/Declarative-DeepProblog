# run_mnistr_prototype.py
import argparse, os, pickle, csv, time
from collections.abc import Mapping
from typing import Iterator

import torch
from sklearn.metrics import accuracy_score
from problog.logic import Term, Var, Constant
from torchvision.utils import save_image

from deepproblog.engines import ExactEngine
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.logger import VerboseLogger
from deepproblog.query import Query

# tensor sources you already use
from data import MNIST_train, MNIST_test

# MNIST-R datasets (from data_mnistr.py)
from data_mnistr import (
    MNISTNot34Binary,   # not_3_or_4(Image) -> {0,1}
    MNISTCount3s,       # count_digit_3([Imgs]) -> Count
    MNISTCount34,       # count_3_or_4([Imgs]) -> Count
    MNISTLessThanBinary # less_than(ImageA, ImageB) -> {0,1}
)

# Helpers: save/load Model state
def load_state(model, path):
    with open(path, 'rb') as f:
        model.__setstate__(pickle.load(f))

def save_state(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model.__getstate__(), f)

# Prototype tensor source (digits 0..9)
class LatentSource(Mapping[Term, torch.Tensor]):
    def __init__(self, nr_embeddings=10, embedding_size=12):
        self.data = torch.nn.Embedding(nr_embeddings, embedding_size)

    def __getitem__(self, index: tuple[Term]) -> torch.Tensor:
        i = torch.LongTensor([int(index[0])])  # 0..9
        return self.data(i)[0]

    def __len__(self): return self.data.num_embeddings
    def __iter__(self) -> Iterator[torch.Tensor]:
        for i in range(len(self)): yield self.data.weight[i]

# CLI
ap = argparse.ArgumentParser("Prototype-based MNIST-R with DeepProbLog")
ap.add_argument("--task", choices=["not34","count3","count34","lessthan"], default="not34")
ap.add_argument("--ae_type", choices=["ae","vae"], default="vae")
ap.add_argument("--model_type", choices=["vae","diffusion"], default="vae")
ap.add_argument("--list_len", type=int, default=5, help="for count tasks")
ap.add_argument("--epochs", type=int, default=10)
ap.add_argument("--batch_size", type=int, default=16)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--inference_only", action=argparse.BooleanOptionalAction, default=False)
ap.add_argument("--save_dir", default="saved_models")
ap.add_argument("--out_dir", default="output")
args = ap.parse_args()

# Data (choose MNIST-R dataset)
if args.task == "not34":
    train_set = MNISTNot34Binary("train", seed=args.seed)
    test_set  = MNISTNot34Binary("test",  seed=args.seed+1)
elif args.task == "count3":
    train_set = MNISTCount3s("train", list_len=args.list_len, seed=args.seed)
    test_set  = MNISTCount3s("test",  list_len=args.list_len, seed=args.seed+1)
elif args.task == "count34":
    train_set = MNISTCount34("train", list_len=args.list_len, seed=args.seed)
    test_set  = MNISTCount34("test",  list_len=args.list_len, seed=args.seed+1)
elif args.task == "lessthan":
    train_set = MNISTLessThanBinary("train", seed=args.seed)
    test_set  = MNISTLessThanBinary("test",  seed=args.seed+1)
else:
    raise ValueError

# Networks (prototype encoder/decoder)
latent_dim = 12
if args.model_type == "vae":
    from networks.VAE_networks import encoder, decoder
else:
    from networks.DDPM_networks import encoder, decoder

encoder_net, enc_opt = encoder(latent_dim)
decoder_net, dec_opt = decoder(latent_dim)

enc = Network(encoder_net, "encoder"); enc.optimizer = enc_opt
dec = Network(decoder_net, "decoder"); dec.optimizer = dec_opt

# Program & model
prog_path = "models/prototype_vae_mnistr.pl"  # provided below
with open(prog_path) as f:
    program_string = f.read()

logger = VerboseLogger(log_every=100)
model = Model(program_string, [enc, dec], logger=logger)
engine = ExactEngine(model, cache_memory=True)

# tensor sources for images (as in your existing code)
model.add_tensor_source("train", MNIST_train)
model.add_tensor_source("test",  MNIST_test)

# prototypes
emb_dim = latent_dim*2 if args.ae_type == "vae" else latent_dim
proto = LatentSource(nr_embeddings=10, embedding_size=emb_dim)

state_file = os.path.join(args.save_dir, f"mnistr_proto_{args.model_type}_{args.task}.pkl")
proto_file = os.path.join(args.save_dir, f"{args.model_type}_latent_source_prototype_digits.torch")

if args.inference_only and os.path.exists(state_file):
    # restore model and prototypes
    load_state(model, state_file)
    with open(proto_file, 'rb') as f:
        proto = pickle.load(f)
    model.add_tensor_source("prototype_digit", proto)
else:
    # (re)train
    model.add_tensor_source("prototype_digit", proto)
    print(f"Training task={args.task} | epochs={args.epochs} | batch={args.batch_size} | ae={args.ae_type}")
    model.fit(dataset=train_set, engine=engine, batch_size=args.batch_size, shuffle=True, stop_condition=args.epochs)

    # save
    os.makedirs(args.save_dir, exist_ok=True)
    with open(proto_file, 'wb') as f:
        pickle.dump(model.tensor_sources["prototype_digit"], f)
    save_state(model, state_file)

# Evaluate
y_pred = model.predict(dataset=test_set, engine=engine)
y_true = test_set.get_labels().numpy()
acc = accuracy_score(y_true, y_pred)
print(f"Test accuracy: {acc:.4f}")

# Small demo query per task
from deepproblog.query import Query
if args.task == "not34":
    q = Query(Term('not_3_or_4', Term('tensor', Term('test', Constant(0))), Var('Y')))
elif args.task == "count3":
    # build a tiny list [p0,p1,p2]
    L = [Term("p0"), Term("p1"), Term("p2")]
    subs = {
        L[0]: Term("tensor", Term("test", Constant(0))),
        L[1]: Term("tensor", Term("test", Constant(1))),
        L[2]: Term("tensor", Term("test", Constant(2))),
    }
    q = Query(Term('count_digit_3', Term('.', L[0], Term('.', L[1], Term('.', L[2], Term('[]')))), Var('C')), subs)
elif args.task == "count34":
    L = [Term("p0"), Term("p1"), Term("p2")]
    subs = {
        L[0]: Term("tensor", Term("test", Constant(3))),
        L[1]: Term("tensor", Term("test", Constant(4))),
        L[2]: Term("tensor", Term("test", Constant(5))),
    }
    q = Query(Term('count_3_or_4', Term('.', L[0], Term('.', L[1], Term('.', L[2], Term('[]')))), Var('C')), subs)
else:  # lessthan
    XA, XB = Term("xa"), Term("xb")
    subs = {
        XA: Term("tensor", Term("test", Constant(0))),
        XB: Term("tensor", Term("test", Constant(1))),
    }
    q = Query(Term('less_than', XA, XB, Var('Y')), subs)

ans = model.query(q, engine).result
print("Example query answers:", ans)
