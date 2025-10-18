import argparse, os, pickle, csv, time
from collections.abc import Mapping
from typing import Iterator
from problog.logic import Term, Var, Constant
from torchvision.utils import save_image

import torch
from sklearn.metrics import accuracy_score

from deepproblog.engines import ExactEngine
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.logger import VerboseLogger
from deepproblog.query import Query

from data import HWFDataset, hwf_images

# -----------------------
# Utilities
# -----------------------
def load_state(model, state_file):
    with open(state_file, 'rb') as f:
        state_dict = pickle.load(f)
    model.__setstate__(state_dict)

def save_state(model, state_file):
    state_dict = model.__getstate__()
    os.makedirs(os.path.dirname(state_file), exist_ok=True)
    with open(state_file, 'wb') as f:
        pickle.dump(state_dict, f)

class LatentSource(Mapping[Term, torch.Tensor]):
    """Prototype storage as a tensor source."""
    def __init__(self, nr_embeddings=10, embedding_size=12) -> None:
        super().__init__()
        self.data = torch.nn.Embedding(nr_embeddings, embedding_size)

    def __getitem__(self, index: tuple[Term]) -> torch.Tensor:
        # index is a tuple of Prolog Terms; the first is the class symbol
        # map '+,-,*,/' to indices for operators
        key = index[0]
        if isinstance(key, str):
            s = key
        else:
            s = str(key)  # Term -> "+", "-", "*", "/"
        if s in {"+", "-", "*", "/"}:
            mapping = {"+": 0, "-": 1, "*": 2, "/": 3}
            i = torch.LongTensor([mapping[s]])
        else:
            # digits: ensure it’s an int 0..9
            i = torch.LongTensor([int(s)])
        return self.data(i)[0]

    def __len__(self) -> int:
        return self.data.num_embeddings

    def __iter__(self) -> Iterator[torch.Tensor]:
        for i in range(len(self)):
            yield self.data.weight[i]

# -----------------------
# Args / config
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HWF dual-prototype (digits & ops) with DeepProbLog")
    parser.add_argument("--ae_type", type=str, default="vae", choices=["ae", "vae"])
    parser.add_argument("--model_type", type=str, default="vae", choices=["vae", "diffusion"])
    parser.add_argument("--save_path", type=str, default="")
    parser.add_argument("--inference_only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--show_all", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--N", type=int, default=1, help="Figure size for HWF")
    parser.add_argument("--curriculum", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    ae_type         = args.ae_type
    model_type      = args.model_type
    inference_only  = args.inference_only
    show_all        = args.show_all
    N               = args.N
    curriculum      = args.curriculum

    name = f"hwf_proto_{model_type}_N{N}"
    output_path = f"output/{args.save_path}".rstrip("/") + ("/" if args.save_path else "")
    os.makedirs(output_path, exist_ok=True)

    # -----------------------
    # Data
    # -----------------------
    try:
        if curriculum:
            train_set = HWFDataset("train2", lambda x: x <= N)
            val_set   = HWFDataset("val",    lambda x: x <= N)
            test_set  = HWFDataset("test",   lambda x: x <= N)
        else:
            train_set = HWFDataset("train2", lambda x: x == N)
            val_set   = HWFDataset("val",    lambda x: x == N)
            test_set  = HWFDataset("test",   lambda x: x == N)
    except FileNotFoundError:
        raise SystemExit("HWF dataset not found. See README to download it.")

    # -----------------------
    # Networks (45×45 HWF)
    # -----------------------
    if model_type == "vae":
        from networks.HWF_VAE_networks import encoder, decoder
    elif model_type == "DDPM":
        from networks.HWF_DDPM_networks import encoder, decoder 

    embed_size = 12
    enc_mod, enc_opt = encoder(embed_size)   # returns (module, optimizer)
    dec_mod, dec_opt = decoder(embed_size)

    enc = Network(enc_mod, "encoder"); enc.optimizer = enc_opt
    dec = Network(dec_mod, "decoder"); dec.optimizer = dec_opt

    # -----------------------
    # Program & Model
    # -----------------------
    # This program is the one you pasted (two families: prototype_digit/op, encoder, decoder)
    prefix = "inference_" if inference_only else ""
    program_path = f"models/{prefix}prototype_hwf.pl"  # <- save your Prolog there
    with open(program_path) as f:
        program_string = f.read()

    logger = VerboseLogger(log_every=10)
    model = Model(program_string, [enc, dec], logger=logger)
    engine = ExactEngine(model, cache_memory=True)

    # Tensor sources: images + two prototype families
    model.add_tensor_source("hwf", hwf_images)

    # Files
    model_state_file = f"saved_models/hwf_{model_type}.pkl"
    proto_digit_file = f"saved_models/{model_type}_latent_source_prototype_digit.torch"
    proto_op_file    = f"saved_models/{model_type}_latent_source_prototype_op.torch"

    # -----------------------
    # Train / Load
    # -----------------------
    if inference_only and os.path.exists(model_state_file):
        # restore model
        load_state(model, model_state_file)

        # restore both prototype families
        with open(proto_digit_file, 'rb') as f:
            latent_digit = pickle.load(f)
        with open(proto_op_file, 'rb') as f:
            latent_op = pickle.load(f)

        model.add_tensor_source('prototype_digit', latent_digit)
        model.add_tensor_source('prototype_op', latent_op)

    else:
        latent_digit = LatentSource(nr_embeddings=10, embedding_size=embed_size*2)  # mean+std
        latent_op    = LatentSource(nr_embeddings=4,  embedding_size=embed_size*2)

        model.add_tensor_source('prototype_digit', latent_digit)
        model.add_tensor_source('prototype_op', latent_op)

        print(f"Training HWF: N={N}, curriculum={curriculum}, ae={ae_type}, model={model_type}")
        model.fit(dataset=train_set, engine=engine, batch_size=16, shuffle=True, stop_condition=30)

        # save everything
        with open(proto_digit_file, 'wb') as f:
            pickle.dump(model.tensor_sources['prototype_digit'], f)
        with open(proto_op_file, 'wb') as f:
            pickle.dump(model.tensor_sources['prototype_op'], f)
        save_state(model, model_state_file)

        # quick val/test (whatever your dataset’s labels represent)
        y_val_pred = model.predict(dataset=val_set, engine=engine)
        y_val_true = val_set.get_labels().numpy()
        print("Val accuracy:\t", accuracy_score(y_val_true, y_val_pred))

        y_test_pred = model.predict(dataset=test_set, engine=engine)
        y_test_true = test_set.get_labels().numpy()
        print("Test accuracy:\t", accuracy_score(y_test_true, y_test_pred))

    # -----------------------
    # Optional: freeze for pure inference
    # -----------------------
    for p in model.networks['encoder'].parameters():
        p.requires_grad = False
    for p in model.networks['decoder'].parameters():
        p.requires_grad = False
    for p in model.tensor_sources['prototype_digit'].data.parameters():
        p.requires_grad = False
    for p in model.tensor_sources['prototype_op'].data.parameters():
        p.requires_grad = False

    # -----------------------
    # Example query (adapt to your program)
    # -----------------------
    # classify a single digit image to "9"
    query = Query(Term('detect_number', Var('X'), Constant(9)))
    answers = model.query(query, engine).result
    print(f"{answers=}")

    # Save the top grounding image for inspection
    top = max(answers, key=lambda k: answers[k])
    if len(top.args) == 2:
        tensor_term, label = top.args
        tensor_im = model.get_tensor(tensor_term).detach()
        out_path = os.path.join(output_path, f'{tensor_term}_term.png')
        save_image(tensor_im, out_path, value_range=(-1.0, 1.0))
        print("Saved:", out_path)
