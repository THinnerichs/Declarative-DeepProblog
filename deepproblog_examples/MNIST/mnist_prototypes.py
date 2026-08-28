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


# -----------------------
# Utilities
# -----------------------
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


# -----------------------
# LatentSource for prototypes
# -----------------------
class LatentSource(Mapping[Term, torch.Tensor]):
    """
    Tensor source that returns a learnable latent vector per class index.
    Accessed by terms like tensor(prototype, Constant(i)).
    """

    def __init__(self, nr_embeddings=10, embedding_size=12) -> None:
        super().__init__()
        self.data = torch.nn.Embedding(nr_embeddings, embedding_size)

    def __getitem__(self, index: tuple[Term]) -> torch.Tensor:
        # index = (Constant(i),) or something equivalent
        i = torch.tensor([int(index[0])], dtype=torch.long)
        return self.data(i)[0]

    def __len__(self) -> int:
        return int(self.data.num_embeddings)

    def __iter__(self) -> Iterator[torch.Tensor]:
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
    prefix = "inference_" if args.inference_only else ""
    suffix = "_mnistr" if args.problem in ["not34", "count3", "count34", "lessthan", "sum2", "sum3", "sum4"] else ""
    program_path = f"models/{prefix}prototype_vae{suffix}.pl"
    with open(program_path) as f:
        program_string = f.read()

    logger = VerboseLogger(log_every=100)
    model = Model(program_string, [enc, dec], logger=logger)
    engine = build_engine(model, args.engine)


    # Latents
    emb_dim = embed_size * 2
    latent = LatentSource(embedding_size=emb_dim, nr_embeddings=10)
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


def run_rq3_2(model, engine, test_set, *, max_groundings: int = 100, seed: int = 0):
    """
    Mask-all regeneration metric on the *actual* test_set.
    For each sampled query:
      - replace every image substitution with a fresh Var,
      - query the model, take MAP grounding (top-1),
      - for each image slot, compare NN(gen) vs NN(gt),
      - a query counts as correct only if *all* slots match.

    Returns: (accuracy, successes)
      accuracy  : float in [0,1]
      successes : list of test indices that were fully correct
    """
    rng = random.Random(seed)
    num_items = len(test_set)
    if num_items == 0:
        return 0.0, []

    # we’ll evaluate at most `max_groundings` queries (and 1 grounding per query)
    max_queries = min(max_groundings, num_items)

    successes = []
    evaluated = 0
    correct = 0

    while evaluated < max_queries:
        # pick an index (DeepProbLog datasets usually 1-based in to_query, but many of your
        # datasets use standard 0-based __getitem__; we rely on the dataset API you already use)
        idx = rng.randint(1, num_items)  # consistent with your addition dataset’s to_query

        query = test_set.to_query(idx)

        # 1) figure out placeholder order (digit slots) from the term,
        #    fallback to sorted substitution keys if needed
        placeholders = _collect_vars_in_order(query.term)
        if not placeholders:
            placeholders = sorted(list(query.substitution.keys()), key=lambda t: str(t))

        # 2) store GT tensor terms (we’ll NN-label them later)
        gt_terms_by_pos = [query.substitution[p] for p in placeholders]

        # 3) mask *all* image slots: substitute each tensor term by a fresh Var
        new_subs = {}
        for p in placeholders:
            new_subs[p] = Var(str(p).upper())
        # keep other bindings (e.g., sums) intact
        for k, v in query.substitution.items():
            if k not in new_subs:
                new_subs[k] = v
        query.substitution = new_subs

        # 4) query model; limit to top-1 grounding → we process 1 grounding per query
        answers = model.query(query, engine).result
        if not answers:
            evaluated += 1
            continue
        best = argmax_answer(answers)

        # 5) extract generated tensors in order and compare labels slot-wise
        gen_terms = _collect_tensors_in_order(best)
        if len(gen_terms) < len(placeholders):
            # conservative skip if we can’t align slots
            evaluated += 1
            continue

        all_ok = True
        for pos in range(len(placeholders)):
            gen_tensor = model.get_tensor(gen_terms[pos]).detach()
            gt_tensor  = model.get_tensor(gt_terms_by_pos[pos]).detach()

            nn_gen = nearest_neighbor_label(gen_tensor, test_set)
            nn_gt  = nearest_neighbor_label(gt_tensor,  test_set)

            if nn_gen != nn_gt:
                all_ok = False
                break

        if all_ok:
            correct += 1
            successes.append(idx)

        evaluated += 1

    acc = correct / evaluated if evaluated > 0 else 0.0
    return acc, successes


# Run RQ3_2 multi_add complex declarative queries.
def run_multiadd4_rq3_2(model: Model, engine, number_len: int = 4, values_to_mask: int = 4, n: int = 100, seed: int = 42):
    """
    Construct an addition dataset with two numbers of length `number_len`,
    mask exactly `values_to_mask` digit positions (across both operands),
    regenerate with the model, and compute generative accuracy.
    """
    import random

    rng = random.Random(seed)
    ds = addition(number_len, "test", seed=seed)

    ok = 0
    for _ in range(n):
        # 1) pick a sample and get its Query
        idx = rng.randint(1, len(ds))           # MNIST datasets often use 1-based indices
        query = ds.to_query(idx)                 # Query(Term('addition', <args...>), subs)

        # 2) Get placeholders IN ORDER from the query structure itself
        placeholders_ordered = _collect_vars_in_order(query.term)
        if len(placeholders_ordered) != 2 * number_len:
            # Fallback: if dataset differs, derive from substitution dict order (less safe)
            placeholders_ordered = sorted(list(query.substitution.keys()), key=lambda t: str(t))

        # 3) Choose which positions to mask (exactly `values_to_mask`)
        mask_positions = rng.sample(range(len(placeholders_ordered)), k=values_to_mask)
        masked_vars = [placeholders_ordered[i] for i in sorted(mask_positions)]

        # 4) Save ground-truth tensors for the masked vars, then replace them with fresh logic vars
        gt_terms_by_var = {v: query.substitution[v] for v in masked_vars}
        new_subs = {}
        for v in placeholders_ordered:
            if v in masked_vars:
                new_subs[v] = Var(str(v).upper())   # turn into fresh variable
            else:
                new_subs[v] = query.substitution[v]
        # keep other (non-digit) bindings intact (e.g., sum), if present
        for k, v in query.substitution.items():
            if k not in new_subs:
                new_subs[k] = v
        query.substitution = new_subs

        # 5) Query the model and take MAP grounding
        answers = model.query(query, engine).result
        if not answers:
            continue
        best = max(answers, key=lambda k: answers[k])

        # 6) From the best grounding, collect ALL tensor(...) terms in-order
        grounded_tensors_ordered = _collect_tensors_in_order(best)

        # Sanity: grounded list should have same length as placeholders order
        if len(grounded_tensors_ordered) < len(placeholders_ordered):
            # Some programs may only materialize newly generated tensors.
            # In that case, we try to map by going through masked slots first.
            # Here we just skip this example to keep metric conservative.
            continue

        # 7) Compare NN labels (generated vs ground truth) at masked positions
        all_correct = True
        for pos in mask_positions:
            var = placeholders_ordered[pos]
            gen_term = grounded_tensors_ordered[pos]   # tensor(...) at that slot
            gen_tensor = model.get_tensor(gen_term).detach()

            gt_term = gt_terms_by_var[var]
            gt_tensor = model.get_tensor(gt_term).detach()

            nn_gen = nearest_neighbor_label(gen_tensor, ds)
            nn_gt  = nearest_neighbor_label(gt_tensor, ds)

            if nn_gen != nn_gt:
                all_correct = False
                break

        if all_correct:
            ok += 1

    return ok / n if n > 0 else 0.0


def extract_generated_terms_for_vars(best_key: Term, masked_vars: List[Term]) -> Dict[Term, Term]:
    """
    Try to pull, from the best grounding term, the tensor terms that correspond
    to our masked variables. We rely on variable-name order.
    This helper is intentionally conservative and should work with the typical
    addition list structure you’re using.
    """
    # Flatten all tensor(...) terms in 'best_key' in order of appearance:
    collected: List[Term] = []
    def walk(t):
        if isinstance(t, Term):
            if t.functor == "tensor":
                collected.append(t)
            for a in t.args:
                walk(a)

    walk(best_key)

    # Heuristic: the first len(masked_vars) tensor terms that *aren't* part of the unmasked inputs
    # correspond to generated ones. Since we masked a whole operand of length K, grab K items.
    k = len(masked_vars)
    if len(collected) < k:
        # fall back to the entire list (best effort)
        k = len(collected)
    mapping = {masked_vars[i]: collected[i] for i in range(k)}
    return mapping


if __name__ == "__main__":
    main()
