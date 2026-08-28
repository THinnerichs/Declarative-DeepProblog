# run_mnist_prototypes.py
import argparse
import os
import pickle
import csv
import time
import random
from collections.abc import Mapping
from typing import Iterator, Tuple, Dict, Any, List

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from problog.logic import Term, Var, Constant
from torchvision.utils import save_image

from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.logger import VerboseLogger
from deepproblog.engines import ExactEngine, ApproximateEngine

# local imports (unchanged)
from data import MNIST, addition, MNIST_train, MNIST_test
from deepproblog.query import Query

from torchvision.utils import save_image  # already imported at top, just here for clarity
import torch

def sample_latent_from_proto(proto_vec: torch.Tensor) -> torch.Tensor:
    """
    proto_vec: 1D tensor of size 2 * latent_dim = [mu, logvar].
    Returns: sampled latent z of size latent_dim.
    """
    mu, logvar = torch.chunk(proto_vec, 2, dim=0)
    eps = torch.randn_like(mu)
    z = mu + eps * torch.exp(0.5 * logvar)
    return z


def save_prototype_samples(
    latent: "LatentSource",
    decoder_network: torch.nn.Module,
    out_dir: str,
    n_protos_per_digit: int = 3,
    n_digits: int = 10,
):
    """
    For each digit C in [0..n_digits-1] and prototype index K in [0..n_protos_per_digit-1],
    sample a latent from prototype (C,K), decode it, and save an image.
    """
    os.makedirs(out_dir, exist_ok=True)
    decoder_network.eval()

    # infer device from decoder parameters (in case you move model to GPU)
    device = next(decoder_network.parameters()).device

    with torch.no_grad():
        for c in range(n_digits):
            for k in range(n_protos_per_digit):
                idx = c * n_protos_per_digit + k
                proto_vec = latent.data.weight[idx].to(device)  # shape: (2 * latent_dim,)

                z = sample_latent_from_proto(proto_vec)         # shape: (latent_dim,)
                z = z.unsqueeze(0)                              # [1, latent_dim]
                x = decoder_network(z)                          # [1, 1, H, W] typically in [-1,1]

                # map from [-1,1] back to [0,1] for saving
                x01 = x * 0.5 + 0.5

                fname = os.path.join(out_dir, f"prototype_{c}_{k}.png")
                save_image(x01, fname)
                print(f"[proto] saved sample for class {c}, proto {k} -> {fname}")

def set_seed(seed: int | None):
    if seed is None:
        return
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_pickle(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


# ----------------------------------------------------------
# LatentSource for (C,K) prototypes: C in [0..9], K in [0..2]
# ----------------------------------------------------------
class LatentSource(Mapping):
    """
    Tensor source that returns a learnable latent vector per prototype (C,K).
    Expected access pattern in Prolog: tensor(prototype(C,K))

    We flatten (C,K) -> idx = C * n_protos_per_class + K and store
    all embeddings in a single nn.Embedding.
    """

    def __init__(self, nr_classes: int = 10, n_protos_per_class: int = 3, embedding_size: int = 12) -> None:
        super().__init__()
        self.nr_classes = nr_classes
        self.n_protos_per_class = n_protos_per_class
        self.data = torch.nn.Embedding(nr_classes * n_protos_per_class, embedding_size)

    # ---- internal helper -------------------------------------------------
    def _index_from_term(self, index) -> int:
        """
        index is the tuple passed by DeepProbLog for tensor(prototype(...)).

        We try to be robust to a couple of shapes:
          - index = (Term('prototype', [C, K]),)
          - index = (C, K)
          - index = (C,)   (fallback to K=0 if ever needed)
        """
        if len(index) == 1:
            term = index[0]
            # Case: prototype(C,K) as a structured Term
            if hasattr(term, "arity") and term.arity == 2:
                c = int(term.args[0])
                k = int(term.args[1])
            else:
                # Fallback: just a single digit index
                c = int(term)
                k = 0
        elif len(index) == 2:
            # Case: tensor(prototype, C, K) style
            c = int(index[0])
            k = int(index[1])
        else:
            raise ValueError(f"Unexpected prototype index: {index!r}")

        if not (0 <= c < self.nr_classes and 0 <= k < self.n_protos_per_class):
            raise IndexError(f"Prototype index out of range: C={c}, K={k}")

        return c * self.n_protos_per_class + k

    # ---- Mapping interface -----------------------------------------------
    def __getitem__(self, index) -> torch.Tensor:
        flat_idx = self._index_from_term(index)
        i = torch.tensor([flat_idx], dtype=torch.long)
        return self.data(i)[0]

    def __len__(self) -> int:
        return int(self.data.num_embeddings)

    def __iter__(self):
        for i in range(len(self)):
            yield self.data.weight[i]

# -----------------------
# CLI parse
# -----------------------
def parse_args():
    p = argparse.ArgumentParser("Prototype-based MNIST DPL runner")
    p.add_argument("--model_type", choices=["vae", "diffusion"], default="vae")
    p.add_argument("--problem", choices=["digit", "addition", "not34", "count3", "count34", "lessthan", "sum2", "sum3", "sum4"], default="digit")
    p.add_argument(
        "--list_len",
        type=int,
        default=5,
        help="List length for counting tasks (ignored for digit/addition/sum2/3/4 which are fixed)",
    )
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--inference_only", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--show_all", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--save_path", type=str, default="")
    p.add_argument("--engine", choices=["exact", "approximate"], default="exact")

    # RQ blocks
    p.add_argument("--run_rq3_1", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--run_rq3_2", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--rq3_2_n", type=int, default=100, help="number of masked queries")
    p.add_argument("--rq3_2_len", type=int, default=4, help="length of numbers for addition masking")

    return p.parse_args()


# -----------------------
# Build encoder/decoder
# -----------------------
def build_enc_dec(model_type: str, latent_dim: int = 12):
    if model_type == "vae":
        from networks.VAE_networks import encoder, decoder
    else:
        from networks.DDPM_networks import encoder, decoder
    encoder_network, enc_opt = encoder(latent_dim)
    decoder_network, dec_opt = decoder(latent_dim)
    return encoder_network, enc_opt, decoder_network, dec_opt


# -----------------------
# Engine
# -----------------------
def build_engine(model: Model, name: str):
    if name == "exact":
        return ExactEngine(model, cache_memory=True)
    else:
        return ApproximateEngine(
            model,
            1,
            ApproximateEngine.geometric_mean,
            timeout=30,
            ignore_timeout=True,
            exploration=True,
        )


# -----------------------
# RQ helpers
# -----------------------
def argmax_answer(answers):
    return max(answers, key=lambda k: answers[k])

def nearest_neighbor_label(tensor1: torch.Tensor, dataset) -> int:
    """Return label of nearest training image to `tensor1` by L2 distance."""
    best_y, best_dist = None, float("inf")
    for im, y in dataset.data:
        dist = torch.norm(tensor1 - im)
        if dist < best_dist:
            best_dist = dist
            best_y = int(y)
    return best_y


def map_mask_positions(subs: Dict[Any, Any], keys_to_mask: list[Term]) -> Dict[Term, Term]:
    """
    Given original substitution dict (var -> tensor(term(...))) and the set of
    vars we masked, build an ordered mapping from each masked var -> the slot it occupies.
    """
    # In DPL, subs keys are Term placeholders (e.g., p0_0, p0_1 ...), values are tensor terms.
    # We simply keep the same variable names; the grounding key in query result will contain
    # the newly generated tensors in the same structure.
    return {k: k for k in keys_to_mask}


def argmax_answer(answers: Dict[Term, float]) -> Term:
    """Return the grounding with max probability."""
    return max(answers, key=lambda k: answers[k])


# -----------------------
# Main
# -----------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    # Names & paths
    method = "exact" if args.engine == "exact" else "approximate"
    name = f"{args.problem}_{args.model_type}_{method}"
    out_dir = os.path.join("output", args.problem, args.save_path)
    os.makedirs(out_dir, exist_ok=True)

    # Data
    if args.problem == "digit":
        train_set = MNIST("train")
        test_set = MNIST("test")
    elif args.problem == "addition":
        N = 1
        train_set = addition(N, "train")
        test_set = addition(N, "test")
    elif args.problem == "not34":
        train_set = MNISTNot34Binary("train", seed=args.seed)
        test_set  = MNISTNot34Binary("test",  seed=args.seed + 1)
    elif args.problem == "count3":
        train_set = MNISTCount3s("train", list_len=args.list_len, seed=args.seed)
        test_set  = MNISTCount3s("test",  list_len=args.list_len, seed=args.seed + 1)
    elif args.problem == "count34":
        train_set = MNISTCount34("train", list_len=args.list_len, seed=args.seed)
        test_set  = MNISTCount34("test",  list_len=args.list_len, seed=args.seed + 1)
    elif args.problem == "lessthan":
        train_set = MNISTLessThanBinary("train", seed=args.seed)
        test_set  = MNISTLessThanBinary("test",  seed=args.seed + 1)
    elif args.problem == "sum2":
        train_set = MNISTSum2("train", seed=args.seed)
        test_set  = MNISTSum2("test",  seed=args.seed + 1)
    elif args.problem == "sum3":
        train_set = MNISTSum3("train", seed=args.seed)
        test_set  = MNISTSum3("test",  seed=args.seed + 1)
    elif args.problem == "sum4":
        train_set = MNISTSum4("train", seed=args.seed)
        test_set  = MNISTSum4("test",  seed=args.seed + 1)
    else:
        raise ValueError("Unknown task")

    # Build networks
    embed_size = 12
    enc_net, enc_opt, dec_net, dec_opt = build_enc_dec(args.model_type, embed_size)
    enc = Network(enc_net, "encoder"); enc.optimizer = enc_opt
    dec = Network(dec_net, "decoder"); dec.optimizer = dec_opt

    # Load program
    program_path = f"models/n_prototype_vae.pl"
    with open(program_path) as f:
        program_string = f.read()

    logger = VerboseLogger(log_every=100)
    model = Model(program_string, [enc, dec], logger=logger)
    engine = build_engine(model, args.engine)


    # Latents
    # latent = LatentSource(embedding_size=emb_dim, nr_embeddings=10)
    # model.add_tensor_source("prototype", latent)
    emb_dim = embed_size * 2
    n_protos_per_digit = 3 
    latent = LatentSource(
        nr_classes=10,
        n_protos_per_class=n_protos_per_digit,
        embedding_size=emb_dim,
    )
    model.add_tensor_source("prototype", latent)


    # State paths
    model_state = f"saved_models/{args.problem}_{args.model_type}_model.pkl"
    latent_state = f"saved_models/{args.problem}_{args.model_type}_latent_source_prototype.torch"

    # Train or restore
    if args.inference_only and os.path.isfile(model_state) and os.path.isfile(latent_state):
        model.__setstate__(load_pickle(model_state))
        loaded_latent = load_pickle(latent_state)
        model.tensor_sources["prototype"] = loaded_latent
        print(f"[restore] loaded model and prototype")

        # Tensor sources for images + prototypes
        model.add_tensor_source("train", MNIST_train)
        model.add_tensor_source("test", MNIST_test)
    else:
        print(f"[train] epochs={args.epochs}, batch={args.batch_size}")

        # Tensor sources for images + prototypes
        model.add_tensor_source("train", MNIST_train)
        model.add_tensor_source("test", MNIST_test)

        model.fit(
            dataset=train_set,
            engine=engine,
            batch_size=args.batch_size,
            shuffle=True,
            stop_condition=args.epochs,
        )
        save_pickle(model.__getstate__(), model_state)
        save_pickle(model.tensor_sources["prototype"], latent_state)

        # quick test acc
        y_pred = model.predict(dataset=test_set, engine=engine)
        y_true = test_set.get_labels().numpy()
        acc = accuracy_score(y_true, y_pred)
        print("Test accuracy:\t", acc)

        # Optional CSV logging
        csv_name = f'{name}_RQ1.csv'
        with open(csv_name, "a", newline="") as f:
            csv.writer(f).writerow([acc])

    # Freeze for inference (as you did)
    for p in model.networks["encoder"].parameters(): p.requires_grad = False
    for p in model.networks["decoder"].parameters(): p.requires_grad = False
    for p in latent.data.parameters(): p.requires_grad = False

    # Save one sampled image per prototype (C,K)
    proto_out_dir = os.path.join(out_dir, "prototypes")
    save_prototype_samples(
        latent=latent,
        decoder_network=dec_net,
        out_dir=proto_out_dir,
        n_protos_per_digit=n_protos_per_digit,
        n_digits=10,
    )


    # Demo query 
    q = Query(Term("addition", Var("X"), Var("Y"), Constant(9)))
    answers = model.query(q, engine).result
    print("addition(_,_,9) MAP:\n", argmax_answer(answers))

    # RQ3_1: Generative accuracy
    if args.run_rq3_1:
        ut_dir = os.path.join(out_dir, "rq3_1_all")  # or your preferred dir
        q_digit = Query(Term('digit', Var('X'), Var('Y')))
        answers = model.query(q_digit, engine).result
        save_all_groundings(answers, model=model, train_set=train_set, out_dir=out_dir, prefix="digit_all")

        # To enumerate *all* addition groundings, use all vars (X,Y,Z)
        q_add = Query(Term('addition', Var('X'), Var('Y'), Var('Z')))
        answers = model.query(q_add, engine).result
        save_all_groundings(answers, model=model, train_set=train_set, out_dir=out_dir, prefix="addition_all")

    # RQ3_2: mask K digits in two 4-digit numbers, regenerate
    if args.run_rq3_2 and args.problem == "addition":
        print("[RQ3_2] running…")
        rq3_2_accuracy = run_multiadd4_rq3_2(
            model,
            engine,
            number_len=args.rq3_2_len, 
            values_to_mask=4,
            n=args.rq3_2_n,
            seed=args.seed,
        )
        print(f"[RQ3_2] generative accuracy: {rq3_2_accuracy:.4f}")
    elif args.run_rq3_2:
        acc, successes = run_rq3_2(model, engine, test_set, max_groundings=100, seed=args.seed)
        print(f"[RQ3_2] mask-all generative accuracy: {acc:.4f} over {len(successes)} / ≤100")


# --- helpers ---------------------------------------------------------------
def nearest_neighbor_label(img_tensor: torch.Tensor, dataset) -> int:
    """Return label of nearest training image to `img_tensor` by L2 distance."""
    best_y, best_d = None, float("inf")
    for im, y in dataset.data:
        d = torch.norm(img_tensor - im)
        if d < best_d:
            best_d = d
            best_y = int(y)
    return best_y

def save_all_groundings(answers, *, model, train_set, out_dir, prefix="rq3_1",
                        save_csv=True, compute_nn=True, value_range=(-1.0, 1.0)):
    """
    Save images for ALL groundings (sorted by probability descending).
    Handles:
      - digit:    digit(Tensor, Label)
      - addition: addition(Tensor1, Tensor2, Sum)
    """
    os.makedirs(out_dir, exist_ok=True)
    # items = sorted(answers.items(), key=lambda kv: kv[1], reverse=True)
    items = answers.items()

    csv_rows = []
    csv_path = os.path.join(out_dir, f"{prefix}.csv") if save_csv else None

    for rank, (key, prob) in enumerate(items, start=1):
        args = key.args

        # ---- Digit case: (tensor_term, label) ----
        if len(args) == 2:
            tensor_term, label = args
            img = model.get_tensor(tensor_term).detach()

            # filenames
            stem = f"digit_y{label}"
            fname = os.path.join(out_dir, f"{stem}.png")

            # optional NN label
            nn_lab = nearest_neighbor_label(img, train_set) if compute_nn else None

            save_image(img, fname, value_range=value_range)
            print(f"[digit] rank={rank} p={prob:.4f} y={label} nearest label={nn_lab}-> {fname}")
            if save_csv:
                csv_rows.append(["digit", rank, float(prob), int(label), nn_lab if nn_lab is not None else "NA", fname])

        # ---- Addition case: (tensor_term1, tensor_term2, sum_label) ----
        elif len(args) == 3:
            t1, t2, sum_label = args
            img1 = model.get_tensor(t1).detach()
            img2 = model.get_tensor(t2).detach()

            stem = f"add_sum{sum_label}"
            f1 = os.path.join(out_dir, f"{stem}_1.png")
            f2 = os.path.join(out_dir, f"{stem}_2.png")

            if compute_nn:
                nn1 = nearest_neighbor_label(img1, train_set)
                nn2 = nearest_neighbor_label(img2, train_set)
            else:
                nn1 = nn2 = None

            save_image(img1, f1, value_range=value_range)
            save_image(img2, f2, value_range=value_range)
            print(f"[add] rank={rank} p={prob:.4f} sum={sum_label} -> {f1}, {f2}")
            if save_csv:
                csv_rows.append(["addition", rank, float(prob), int(sum_label),
                                 nn1 if nn1 is not None else "NA",
                                 nn2 if nn2 is not None else "NA", f1, f2])
        else:
            # other arities not supported here
            continue

    if save_csv and csv_rows:
        header_digit    = ["task","rank","prob","label","nn_label","file"]
        header_addition = ["task","rank","prob","sum","nn_label_1","nn_label_2","file1","file2"]
        # write a unified CSV (heterogeneous rows are fine)
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["NOTE: digit rows:", *header_digit])
            for row in csv_rows:
                if row[0] == "digit":
                    w.writerow(["", *row])  # digit row
            w.writerow([])
            w.writerow(["NOTE: addition rows:", *header_addition])
            for row in csv_rows:
                if row[0] == "addition":
                    w.writerow(["", *row])  # addition row
        print(f"[csv] wrote {csv_path}")


def _collect_vars_in_order(term):
    """
    Traverse a DeepProbLog Term (list-structured for size>1) and collect the
    placeholder variable names ('p0_0', ...) in left-to-right order.
    Works for addition(list(list), list(list), Sum) structure.
    """
    ordered = []

    def walk(t):
        from deepproblog.query import list2term  # not strictly needed, but ok
        if isinstance(t, Term):
            # variables are Terms with functor like 'p0_0', 'p1_2', etc.
            # we accept anything that starts with 'p' and has '_'
            if t.arity == 0 and isinstance(t.functor, str) and t.functor.startswith("p") and "_" in t.functor:
                ordered.append(t)
            for a in t.args:
                walk(a)

    walk(term)
    return ordered  # [p0_0, p0_1, ..., p1_0, p1_1, ...] in order


def _collect_tensors_in_order(grounding_term):
    """
    Traverse a *grounded* answer Term and collect all tensor(...) terms
    in left-to-right order. The order mirrors the placeholders order above.
    """
    tensors = []

    def walk(t):
        if isinstance(t, Term):
            if t.functor == "tensor":
                tensors.append(t)
            for a in t.args:
                walk(a)

    walk(grounding_term)
    return tensors  

if __name__ == "__main__":
    main()