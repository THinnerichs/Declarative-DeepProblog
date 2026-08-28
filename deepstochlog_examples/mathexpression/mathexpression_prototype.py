from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from shutil import copy2
from time import time
from typing import Union, Dict, Any

import numpy as np
import torch
from PIL import Image
from torch.optim import Adam

from deepstochlog.network import Network, NetworkStore
from deepstochlog.utils import (
    set_fixed_seed,
    create_run_test_query,
)
from deepstochlog.dataloader import DataLoader
from deepstochlog.model import DeepStochLogModel
from deepstochlog.trainer import DeepStochLogTrainer, print_logger, PrintFileLogger
from deepstochlog.term import Term, List

from mathexpression_data import (
    MathExprDataset,
    operator_word_list,
    operator_list,
    all_symbols_list,
    digit_idx_list,
    operator_idx_list,
    mathexpression_dataset_max_seq_length,
    create_our_splits,
)
from networks.network import SymbolEncoder, SymbolDecoder, ProtoSymbol

root_path = Path(__file__).parent


# ---------------------------------------------------------------------------
#  Checkpointing
# ---------------------------------------------------------------------------


def _extract_stds_from_model(model: torch.nn.Module) -> Dict[str, float]:
    """Recover prior_std and sample_std from ProtoSymbol buffers, with defaults."""
    # prior_logvar: [1,D]
    if hasattr(model, "prior_logvar"):
        plv = model.prior_logvar.view(-1)[0].item()
        prior_std = float(math.sqrt(math.exp(plv)))
    else:
        prior_std = 1.50

    # sample_std: [1,D]
    if hasattr(model, "sample_std"):
        sstd = model.sample_std.view(-1)[0].item()
        sample_std = float(sstd)
    else:
        sample_std = 0.30

    return {"prior_std": prior_std, "sample_std": sample_std}


def save_model_checkpoint(
    store: NetworkStore,
    optimizer: Adam | None,
    ckpt_path: Union[str, Path],
) -> None:
    """
    Save both 'number' and 'operator' networks (weights + key hyperparams) and optimizer.
    """
    ckpt_path = Path(ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    num = store.networks["number"].neural_model
    op = store.networks["operator"].neural_model

    stds = _extract_stds_from_model(num)

    arch: Dict[str, Any] = {
        "emb_dim": getattr(num, "emb_dim", 12),
        "use_decoder": getattr(num, "decoder", None) is not None,
        "n_mc_protos": getattr(num, "n_mc", 1),
        "recon_weight": getattr(num, "recon_weight", 1.0),
        "prior_std": stds["prior_std"],
        "sample_std": stds["sample_std"],
    }

    payload = {
        "version": 1,
        "arch": arch,
        "state": {
            "number": num.state_dict(),
            "operator": op.state_dict(),
        },
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "index_list": {
            "number": [str(t) for t in store.networks["number"].index_list],
            "operator": [str(t) for t in store.networks["operator"].index_list],
        },
    }
    torch.save(payload, ckpt_path)
    print(f"[save_model_checkpoint] Saved to {ckpt_path.resolve()}")


def load_model_checkpoint(
    ckpt_path: Union[str, Path],
    *,
    device: Union[str, torch.device] = "cpu",
) -> tuple[NetworkStore, Dict[str, Any], Dict[str, Any] | None]:
    """
    Recreate networks with the saved architecture and load weights.
    Returns (networks, arch_dict, optimizer_state).
    """
    ckpt_path = Path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    arch = ckpt["arch"]

    # Backward compatible defaults
    prior_std = float(arch.get("prior_std", 1.50))
    sample_std = float(arch.get("sample_std", 0.30))

    networks = load_expression_networks(
        use_decoder=arch["use_decoder"],
        recon_weight=arch["recon_weight"],
        emb_dim=arch["emb_dim"],
        n_mc_protos=arch["n_mc_protos"],
        prior_std=prior_std,
        sample_std=sample_std,
    )

    networks.networks["number"].neural_model.load_state_dict(
        ckpt["state"]["number"], strict=True
    )
    networks.networks["operator"].neural_model.load_state_dict(
        ckpt["state"]["operator"], strict=True
    )

    opt_state = ckpt.get("optimizer", None)
    print(f"[load_model_checkpoint] Loaded from {ckpt_path.resolve()}")
    return networks, arch, opt_state


# ---------------------------------------------------------------------------
#  Prototype visualisation & generative utilities
# ---------------------------------------------------------------------------


def save_mean_decoded_prototypes(
    store: NetworkStore,
    net_name: str = "number",           # "number" or "operator"
    out_dir: Union[str, Path] = "output/prototypes",
    prefix: str = "",
    also_save_grid: bool = False,
    grid_filename: str = "prototypes_grid.png",
) -> None:
    """
    Decode the MEAN prototypes of the given network and save each as a PNG.

    Assumes:
      - store.networks[net_name].neural_model is a ProtoSymbol
      - model.decoder exists and returns images in [-0.5, 0.5] with shape [C,1,H,W]
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    net = store.networks[net_name]
    model = net.neural_model

    if getattr(model, "decoder", None) is None:
        raise RuntimeError(f"{net_name} has no decoder (decoder=None).")

    labels = [str(t) for t in getattr(net, "index_list", [])]
    if not labels or len(labels) != model.n_classes:
        labels = [str(i) for i in range(model.n_classes)]

    model.eval()
    with torch.no_grad():
        # If helper exists, use it; otherwise decode prototypes directly
        if hasattr(model, "_decode_proto_means"):
            imgs = model._decode_proto_means()  # [C,1,H,W]
        else:
            imgs = model.decoder(model.prototypes)  # [C,1,H,W]

        imgs = imgs.clamp(-0.5, 0.5)
        C, _, H, W = imgs.shape

        # Save each class image separately
        for c in range(C):
            arr01 = (imgs[c, 0].cpu().numpy() + 0.5).clip(0.0, 1.0)
            im = Image.fromarray((arr01 * 255).astype("uint8"), mode="L")
            name = f"{prefix}{labels[c]}.png" if prefix else f"{labels[c]}.png"
            im.save(out_dir / name)

        # Optional: grid
        if also_save_grid:
            cols = int(math.ceil(math.sqrt(C)))
            rows = int(math.ceil(C / cols))
            canvas = Image.new("L", (cols * W, rows * H), color=0)
            for idx in range(C):
                r, c = divmod(idx, cols)
                arr01 = (imgs[idx, 0].cpu().numpy() + 0.5).clip(0.0, 1.0)
                tile = Image.fromarray((arr01 * 255).astype("uint8"), mode="L")
                canvas.paste(tile, (c * W, r * H))
            canvas.save(out_dir / grid_filename)

    print(f"[save_mean_decoded_prototypes] Saved {len(labels)} images to {out_dir.resolve()}")


@torch.no_grad()
def _build_symbol_bank_from_mathexpr(
    dset: MathExprDataset,
    kind: str,
    device: Union[str, torch.device] = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build a bank of individual symbol images (digits or operators) from MathExprDataset.

    Returns:
        imgs:   [N,1,H,W] tensor in [-0.5, 0.5] on `device`
        labels: [N] long tensor with class indices:
            - kind="number": 0..9 (note: HWF only has digits 1..9)
            - kind="operator": 0..3 for +, -, *, /
    """
    device = torch.device(device)
    imgs = []
    labels = []

    for sample in dset.dataset:
        img_ids = sample["image_sequence"]
        lab_seq = sample["label_sequence"]
        for img_id, lab_idx in zip(img_ids, lab_seq):
            sym = all_symbols_list[lab_idx]

            if kind == "number" and lab_idx in digit_idx_list:
                # digits are '1'..'9'; '0' does not really appear in HWF
                class_idx = int(sym)
                imgs.append(dset.tensors[img_id].unsqueeze(0))
                labels.append(class_idx)

            elif kind == "operator" and lab_idx in operator_idx_list:
                # map '+','-','*','/' to 0..3
                class_idx = operator_list.index(sym)
                imgs.append(dset.tensors[img_id].unsqueeze(0))
                labels.append(class_idx)

    if not imgs:
        raise RuntimeError(f"No symbols found of kind='{kind}'")

    imgs_t = torch.cat(imgs, dim=0).to(device)
    labels_t = torch.tensor(labels, dtype=torch.long, device=device)
    return imgs_t, labels_t


@torch.no_grad()
def generative_accuracy_protos(
    store: NetworkStore,
    data: MathExprDataset,
    n_samples_per_proto: int = 100,
    device: Union[str, torch.device] = "cpu",
) -> Dict[str, float]:
    """
    Generative accuracy (per your spec):

    For each semantic class (digit / operator) and its prototype:
        - sample n images around the prototype,
        - for each sample, find the closest dataset image (pixelwise MSE),
        - predicted label = label of that nearest neighbour,
        - correct if predicted == original class.

    For digits we only evaluate over classes that are actually present in the data
    (HWF digits are 1..9, no '0').
    """
    device = torch.device(device)
    results: Dict[str, float] = {}

    for kind in ["number", "operator"]:
        net = store.networks[kind].neural_model.to(device)

        bank_imgs, bank_labels = _build_symbol_bank_from_mathexpr(
            data, kind=kind, device=device
        )  # [N,1,H,W], [N]

        # samples: [C_total * n_samples_per_proto,1,H,W]
        samples = net.decode_proto_samples(n_per_class=n_samples_per_proto).to(device)

        bank_flat = bank_imgs.view(bank_imgs.size(0), -1)  # [N,D]
        samp_flat = samples.view(samples.size(0), -1)      # [C_total*n,D]

        classes_present = torch.unique(bank_labels).tolist()
        C_total = net.n_classes
        correct = 0
        total = 0

        for c in classes_present:
            c = int(c)
            if c < 0 or c >= C_total:
                continue  # just in case

            base_idx = c * n_samples_per_proto
            for k in range(n_samples_per_proto):
                idx = base_idx + k
                s = samp_flat[idx : idx + 1]               # [1,D]
                mse = (bank_flat - s).pow(2).mean(dim=1)   # [N]
                nn_idx = mse.argmin().item()
                pred_label = int(bank_labels[nn_idx].item())
                if pred_label == c:
                    correct += 1
                total += 1

        acc = float(correct) / float(total) if total > 0 else 0.0
        results[kind] = acc

    return results


# ---------------------------------------------------------------------------
#  DeepStochLog helpers
# ---------------------------------------------------------------------------


def create_expression_sentence_query(
    number_img: int,
    total_sum: Union[str, float] = "_",
) -> Term:
    """
    Generates query like expression(_, [img1,img2,...]).
    """
    total_sum_term = Term(str(total_sum))
    images_arg = List(*[Term(f"img{i + 1}") for i in range(number_img)])
    return Term("expression", total_sum_term, images_arg)


class GreedyDumbEvaluation:
    """
    Evaluate by greedily classifying every symbol in the context
    and directly evaluating the arithmetic expression.
    """

    def __init__(self, valid_data, test_data, store: NetworkStore):
        self.valid_data = valid_data
        self.test_data = test_data
        self.store = store
        self.header = "Valid acc\tTest acc\t"
        self.max_val = 0.0
        self.test_acc = 0.0

    def _acc(self, data, number_net, operator_net):
        ops = ["+", "-", "*", "/"]
        evaluations = []

        for term in data:
            res_str = ""
            for i, (_, tensor) in enumerate(term.context._context.items()):
                if i % 2 == 0:
                    # digit
                    n = torch.argmax(
                        number_net.neural_model(tensor.unsqueeze(dim=0))
                    ).item()
                    res_str += str(n)
                else:
                    # operator
                    o = torch.argmax(
                        operator_net.neural_model(tensor.unsqueeze(dim=0))
                    ).item()
                    res_str += ops[o]

            try:
                res_val = eval(res_str)
                ground = term.term.arguments[0]
                evaluations.append(int(ground == res_val))
            except Exception:
                evaluations.append(0)

        return float(np.mean(evaluations))

    def __call__(self) -> str:
        number_net = self.store.networks["number"]
        operator_net = self.store.networks["operator"]
        number_net.neural_model.eval()
        operator_net.neural_model.eval()

        valid_acc = self._acc(self.valid_data, number_net, operator_net)
        if valid_acc >= self.max_val:
            self.test_acc = self._acc(self.test_data, number_net, operator_net)
            self.max_val = valid_acc

        number_net.neural_model.train()
        operator_net.neural_model.train()
        return f"{valid_acc}\t{self.test_acc}\t"


# ---------------------------------------------------------------------------
#  Network construction
# ---------------------------------------------------------------------------


def load_expression_networks(
    lr: float = 1e-4,
    use_decoder: bool = True,
    recon_weight: float = 1.0,
    emb_dim: int = 12,
    n_mc_protos: int = 1,
    prior_std: float = 1.50,
    sample_std: float = 0.30,
) -> NetworkStore:
    """
    Create shared encoder/decoder and two ProtoSymbol classifiers
    (digits 0..9 and operators plus/minus/times/div).
    """
    encoder = SymbolEncoder(emb_dim=emb_dim)
    decoder = SymbolDecoder(emb_dim=emb_dim) if use_decoder else None

    number_model = ProtoSymbol(
        encoder=encoder,
        n_classes=10,
        emb_dim=emb_dim,
        decoder=decoder,
        recon_weight=recon_weight,
        n_mc_protos=n_mc_protos,
        bound_variances=False,
        diversity_w=1.0,
        min_ink_coverage=0.02,  # interpreted as min_pixel_var
        coverage_w=1.0,
    )
    operator_model = ProtoSymbol(
        encoder=encoder,
        n_classes=4,
        emb_dim=emb_dim,
        decoder=decoder,
        recon_weight=recon_weight,
        n_mc_protos=n_mc_protos,
        bound_variances=False,
        diversity_w=2.0,
        min_ink_coverage=0.02,
        coverage_w=5.0,
    )
    number_network = Network(
        "number",
        number_model,
        index_list=[Term(str(i)) for i in range(10)],
    )
    operator_network = Network(
        "operator",
        operator_model,
        index_list=[Term(op) for op in operator_word_list],
    )

    return NetworkStore(number_network, operator_network)


# ---------------------------------------------------------------------------
#  Main training / evaluation entry point
# ---------------------------------------------------------------------------


def run(
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 0.003,
    expression_length: int | None = None,
    expression_max_length: int = 3,
    allow_division: bool = True,
    device_str: str | None = None,
    #
    train_size=None,
    test_size=None,
    #
    log_freq: int = 50,
    logger=print_logger,
    test_example_idx=None,
    test_batch_size: int = 100,
    #
    seed: int | None = None,
    verbose: bool = False,
    #
    prior_std: float = 1.50,
    sample_std: float = 0.30,
    do_gen_acc: bool = False,
    save_proto_images: bool = False,
) -> float:
    """
    Train the DeepStochLog model + proto networks for one configuration / seed.
    """
    start_time = time()
    set_fixed_seed(seed)

    networks = load_expression_networks(
        lr=lr,
        use_decoder=True,
        recon_weight=1.0,
        emb_dim=12,
        n_mc_protos=1,
        prior_std=prior_std,
        sample_std=sample_std,
    )

    # Device
    if device_str is not None:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Build query set
    max_length = (
        expression_max_length
        if expression_max_length
        else mathexpression_dataset_max_seq_length
    )
    query = [
        create_expression_sentence_query(number_img=length, total_sum="_")
        for length in range(1, max_length + 1, 2)
    ]

    proving_start = time()
    model = DeepStochLogModel.from_file(
        file_location=str((root_path / "models" / "mathexpression.pl").absolute()),
        query=query,
        networks=networks,
        device=device,
    )
    optimizer = Adam(model.get_all_net_parameters(), lr=lr)
    optimizer.zero_grad()
    proving_time = time() - proving_start
    if verbose:
        logger.print(f"\nProving the program took {proving_time:.2f} seconds")

    # Data
    if expression_length is not None:
        train_data = MathExprDataset(
            split="train",
            num_samples=train_size,
            random_seed=seed,
            allow_division=allow_division,
            expression_length=expression_length,
        )
        valid_data = MathExprDataset(
            split="val",
            num_samples=test_size,
            random_seed=seed,
            allow_division=allow_division,
            expression_length=expression_length,
        )
        test_data = MathExprDataset(
            split="test",
            num_samples=test_size,
            random_seed=seed,
            allow_division=allow_division,
            expression_length=expression_length,
        )
    else:
        train_data = MathExprDataset(
            split="train",
            num_samples=train_size,
            random_seed=seed,
            allow_division=allow_division,
            expression_max_length=expression_max_length,
        )
        valid_data = MathExprDataset(
            split="val",
            num_samples=test_size,
            random_seed=seed,
            allow_division=allow_division,
            expression_max_length=expression_max_length,
        )
        test_data = MathExprDataset(
            split="test",
            num_samples=test_size,
            random_seed=seed,
            allow_division=allow_division,
            expression_max_length=expression_max_length,
        )

    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=len(test_data))

    run_test_query = create_run_test_query(
        model=model,
        test_data=test_data,
        test_example_idx=test_example_idx,
        verbose=verbose,
    )

    greedy_eval = GreedyDumbEvaluation(valid_data, test_data, networks)
    header, acc_fn = greedy_eval.header, greedy_eval

    trainer = DeepStochLogTrainer(
        log_freq=log_freq,
        accuracy_tester=(header, acc_fn),
        logger=logger,
        print_time=verbose,
        test_query=run_test_query,
    )
    trainer.train(
        model=model,
        optimizer=optimizer,
        dataloader=train_dataloader,
        epochs=epochs,
    )

    logger.print("Best val accuracy: " + str(greedy_eval.max_val))
    logger.print("Best test accuracy:" + str(greedy_eval.test_acc))
    logger.print(f"Total time taken: {time() - start_time:.2f} seconds")

    # Optional: generative accuracy on validation symbols
    if do_gen_acc:
        gen_acc = generative_accuracy_protos(
            networks,
            valid_data,
            n_samples_per_proto=100,
            device=device,
        )
        logger.print(
            f"Generative accuracy digits:   {gen_acc['number']:.4f}, "
            f"operators: {gen_acc['operator']:.4f}"
        )

    # Optional: save prototype images
    if save_proto_images:
        save_mean_decoded_prototypes(
            store=networks,
            net_name="number",
            out_dir=root_path / "output/num_means",
            also_save_grid=True,
        )
        save_mean_decoded_prototypes(
            store=networks,
            net_name="operator",
            out_dir=root_path / "output/op_means",
            also_save_grid=True,
        )

    # Save checkpoint
    ckpt_file = root_path / "output" / "proto_model.pt"
    ckpt_file.parent.mkdir(parents=True, exist_ok=True)
    save_model_checkpoint(networks, optimizer, ckpt_file)
    logger.print(f"Saved checkpoint to {ckpt_file}")

    return float(greedy_eval.test_acc)


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ProtoVAE + DeepStochLog for handwritten math expressions"
    )
    parser.add_argument(
        "--inference-only",
        action="store_true",
        help="Do not train; just load checkpoint and run analysis.",
    )
    parser.add_argument(
        "--calc-gen-acc",
        action="store_true",
        help="Calculate generative accuracy (digits + operators).",
    )
    parser.add_argument(
        "--save-proto-images",
        action="store_true",
        help="Save decoded prototype images to output/...",
    )
    parser.add_argument(
        "--prior-std",
        type=float,
        default=1.50,
        help="Latent prior std used in ProtoSymbol.",
    )
    parser.add_argument(
        "--sample-std",
        type=float,
        default=0.30,
        help="Sampling std around prototypes when decoding.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(root_path / "output" / "proto_model.pt"),
        help="Checkpoint path for saving/loading ProtoSymbol weights.",
    )

    args = parser.parse_args()

    # Make sure HWF splits are present
    create_our_splits()

    if args.inference_only:
        # ---------------------------------------------------------------
        #  Inference-only: load trained nets, optionally eval/save stuff
        # ---------------------------------------------------------------
        networks, arch, opt_state = load_model_checkpoint(
            args.checkpoint,
            device="cpu",
        )
        print(
            f"Loaded checkpoint with prior_std={arch.get('prior_std', 'NA')}, "
            f"sample_std={arch.get('sample_std', 'NA')}"
        )

        if args.calc_gen_acc:
            # Use validation split, length 3 (matches the usual training setting)
            valid_data = MathExprDataset(
                split="val",
                num_samples=None,
                random_seed=0,
                allow_division=True,
                expression_length=3,
            )
            gen_acc = generative_accuracy_protos(
                networks,
                valid_data,
                n_samples_per_proto=100,
                device="cpu",
            )
            print(
                f"[inference_only] Generative accuracy digits: {gen_acc['number']:.4f}, "
                f"operators: {gen_acc['operator']:.4f}"
            )

        if args.save_proto_images:
            save_mean_decoded_prototypes(
                store=networks,
                net_name="number",
                out_dir=root_path / "output/num_means",
                also_save_grid=True,
            )
            save_mean_decoded_prototypes(
                store=networks,
                net_name="operator",
                out_dir=root_path / "output/op_means",
                also_save_grid=True,
            )

    else:
        # ---------------------------------------------------------------
        #  Training loop over seeds (as in your original script)
        # ---------------------------------------------------------------
        logs = root_path / "logs_temp"
        logs.mkdir(exist_ok=True)

        # Snapshot the files used for this run
        copy2(__file__, logs)
        copy2(root_path / "mathexpression_data.py", logs)
        copy2(root_path / "models" / "mathexpression.pl", logs)

        for seed in [0, 1, 2, 3, 4]:
            folder = logs / f"{seed}"
            folder.mkdir(exist_ok=True)
            for l in [1]:
                logger = PrintFileLogger(str(folder / f"exact_{l}.txt"))
                _ = run(
                    test_example_idx=0,
                    expression_max_length=l,
                    expression_length=l,
                    epochs=10,
                    batch_size=4,
                    seed=seed,
                    logger=logger,
                    log_freq=100,
                    allow_division=True,
                    verbose=True,
                    device_str="cpu",
                    prior_std=args.prior_std,
                    sample_std=args.sample_std,
                    do_gen_acc=args.calc_gen_acc,
                    save_proto_images=args.save_proto_images,
                )
