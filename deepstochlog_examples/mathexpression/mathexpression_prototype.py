from pathlib import Path
from typing import Union
from shutil import copy2
import torch
import numpy as np
from torch.optim import Adam
from time import time

from pathlib import Path
from PIL import Image
import os

from deepstochlog.network import Network, NetworkStore
from mathexpression_data import (
    MathExprDataset,
    operator_word_list,
    mathexpression_dataset_max_seq_length,
    create_our_splits,
)
from networks.network import SymbolEncoder, SymbolDecoder, ProtoSymbol
from deepstochlog.network import Network, NetworkStore
from deepstochlog.term import Term
from mathexpression_data import operator_word_list

from deepstochlog.utils import (
    calculate_accuracy,
    test_single_instance,
    set_fixed_seed,
    create_model_accuracy_calculator,
    create_run_test_query,
)
from deepstochlog.dataloader import DataLoader
from deepstochlog.model import DeepStochLogModel
from deepstochlog.trainer import DeepStochLogTrainer, print_logger, PrintFileLogger
from deepstochlog.term import Term, List


root_path = Path(__file__).parent

# put in mathexpression_prototype.py (or wherever your saver lives)
from pathlib import Path
from PIL import Image
import torch

# anywhere convenient (e.g., in mathexpression_prototype.py)
from pathlib import Path
from PIL import Image
import torch

def save_number_prototypes(store, out_dir: str = "output", n_samples_per_class: int = 0):
    """
    Saves mean-decoded prototypes as digit_#.png.
    Optionally also saves samples per class as digit_#_sampleK.png.
    Assumes images are in [-0.5, 0.5].
    """
    num = store.networks["number"].neural_model  # ProtoSymbol
    if getattr(num, "decoder", None) is None:
        raise RuntimeError("ProtoSymbol.decoder is None (use_decoder=True required).")
    if not hasattr(num, "prototypes"):
        raise RuntimeError("ProtoSymbol has no 'prototypes' attribute.")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    num.eval()
    with torch.no_grad():
        # --- decode the means (explicit prototypes) ---
        imgs_mu = num.decode_mean_prototypes().cpu()  # [C,1,45,45], in [-0.5,0.5]
        C = imgs_mu.size(0)
        for c in range(C):
            arr = (imgs_mu[c, 0].numpy() + 0.5).clip(0, 1)  # -> [0,1]
            Image.fromarray((arr * 255).astype("uint8"), mode="L").save(out / f"digit_{c}.png")

        # --- optional: samples from N(prototypes, diag(exp(logvar))) ---
        if n_samples_per_class > 0 and hasattr(num, "logvar"):
            std = torch.exp(0.5 * num.logvar)          # [C,D]
            D = std.size(1)
            eps = torch.randn(n_samples_per_class, C, D, device=std.device)
            z = num.prototypes.unsqueeze(0) + eps * std.unsqueeze(0)   # [K,C,D]
            z = z.view(-1, D)
            imgs = num.decoder(z).cpu()                 # [K*C,1,45,45]
            for k in range(n_samples_per_class):
                for c in range(C):
                    idx = k * C + c
                    arr = (imgs[idx, 0].numpy() + 0.5).clip(0, 1)
                    Image.fromarray((arr * 255).astype("uint8"), mode="L").save(
                        out / f"digit_{c}_sample{k}.png"
                    )

    print(f"Saved {C} prototype images to {out.resolve()}")


def save_model_checkpoint(store, optimizer, ckpt_path: str):
    """
    Save both 'number' and 'operator' networks (weights + key hyperparams) and optimizer.
    """
    ckpt_path = Path(ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    num = store.networks["number"].neural_model
    op  = store.networks["operator"].neural_model

    arch = {
        "emb_dim": getattr(num, "emb_dim", 12),
        "use_decoder": num.decoder is not None,
        "n_mc_protos": getattr(num, "n_mc", 1),
        "epsilon_gate": getattr(num, "epsilon_gate", 0.1),
        "recon_weight": getattr(num, "recon_scale", 1.0),
    }

    payload = {
        "version": 1,
        "arch": arch,
        "state": {
            "number": num.state_dict(),
            "operator": op.state_dict(),
        },
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        # (optional) keep the index lists so you can sanity-check mapping
        "index_list": {
            "number": [str(t) for t in store.networks["number"].index_list],
            "operator": [str(t) for t in store.networks["operator"].index_list],
        },
    }
    torch.save(payload, ckpt_path)
    print(f"[save_model_checkpoint] Saved to {ckpt_path.resolve()}")


def load_model_checkpoint(ckpt_path: str, *, device="cpu"):
    """
    Recreate networks with the saved architecture and load weights.
    Returns (networks, arch, optimizer_state).
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    arch = ckpt["arch"]

    # Rebuild networks using your factory so weight sharing is preserved
    networks = load_expression_networks(
        use_decoder=arch["use_decoder"],
        recon_weight=arch["recon_weight"],
        emb_dim=arch["emb_dim"],
        n_mc_protos=arch["n_mc_protos"],
    )

    # Load state dicts
    networks.networks["number"].neural_model.load_state_dict(ckpt["state"]["number"], strict=True)
    networks.networks["operator"].neural_model.load_state_dict(ckpt["state"]["operator"], strict=True)

    opt_state = ckpt.get("optimizer", None)
    print(f"[load_model_checkpoint] Loaded from {Path(ckpt_path).resolve()}")
    return networks, arch, opt_state


def decode_digit_prototypes(store, use_samples: bool = False, sample_seed: int = 0):
    """
    Returns decoded prototype images for the 'number' network as a tensor [10,1,45,45].
    If use_samples=True, decodes a single sample per class from N(prototypes, exp(logvar)).
    Otherwise decodes the means (ProtoSymbol.prototypes).
    """
    num = store.networks["number"].neural_model
    num.eval()
    if getattr(num, "decoder", None) is None:
        raise RuntimeError("ProtoSymbol.decoder is None; cannot decode prototypes.")

    if use_samples and hasattr(num, "logvar"):
        g = torch.Generator(device=num.prototypes.device).manual_seed(sample_seed)
        eps = torch.randn_like(num.prototypes, generator=g)
        z = num.prototypes + eps * torch.exp(0.5 * num.logvar)
        imgs = num.decoder(z)   # [-0.5,0.5], [10,1,45,45]
    else:
        imgs = num.decode_mean_prototypes()  # [-0.5,0.5], [10,1,45,45]
    return imgs


@torch.no_grad()
def generative_accuracy_numbers(store, train_images: torch.Tensor, train_labels: torch.Tensor,
                                use_samples: bool = False, sample_seed: int = 0) -> float:
    """
    Compute generative accuracy as described:
    - For each class c in {0..9}, decode prototype image x_hat_c
    - Find nearest neighbor in the training set by MSE
    - Count correct if the NN's label == c
    Args:
        train_images: [N,1,45,45] in [-0.5,0.5]
        train_labels: [N] LongTensor, values in {0..9}
    """
    # Decode 10 prototype images
    proto_imgs = decode_digit_prototypes(store, use_samples=use_samples, sample_seed=sample_seed)  # [10,1,45,45]

    # Flatten to vectors
    P = proto_imgs.view(10, -1).cpu()       # [10, 2025]
    X = train_images.view(train_images.size(0), -1).cpu()  # [N, 2025]
    y = train_labels.long().cpu()

    # Pairwise MSEs: for each class c, compute MSE(X_n, P_c) over all n
    # Efficient: (X - P_c)^2 mean over dims
    # Build [N,10] table of MSEs
    # (X^2 - 2 X·P + P^2) / D ; but brute-force squared diff is fine for 10 prototypes
    N, D = X.shape
    m = ((X.unsqueeze(1) - P.unsqueeze(0)) ** 2).mean(dim=-1)  # [N,10]

    # For each class c: take argmin over N, check label
    nn_idx = m.argmin(dim=0)   # [10] — index of nearest train image for each class
    preds = y[nn_idx]          # [10]
    correct = (preds == torch.arange(10))
    acc = correct.float().mean().item()
    return acc


def to_uint8(imgs: torch.Tensor) -> np.ndarray:
    """
    imgs: [-0.5,0.5] tensor [N,1,H,W] -> uint8 numpy [N,H,W]
    """
    arr = (imgs.clamp(-0.5, 0.5) + 0.5).mul(255).round().to(torch.uint8).cpu().numpy()
    return arr[:, 0]

def build_digit_bank(dset):
    imgs, labels = [], []
    for i in range(len(dset)):
        x, y = dset[i]                   # x: [1,45,45] in [-0.5,0.5]; y: int 0..9
        imgs.append(x.unsqueeze(0))
        labels.append(int(y))
    return torch.cat(imgs, dim=0), torch.tensor(labels)



def create_expression_sentence_query(number_img: int, total_sum: Union[str, float] = "_"):
    """ Generates sentence query like s(_, [img1,img2,img3,img4,img5,img6,img7], [], _)"""
    total_sum = Term(str(total_sum))
    images_arg = List(*[Term(f"img{i + 1}") for i in range(number_img)])
    return Term("expression", total_sum, images_arg)


class GreedyDumbEvaluation:
    def __init__(self, valid_data, test_data, store):

        self.valid_data = valid_data
        self.test_data = test_data
        self.store = store
        self.header = "Valid acc\tTest acc\t"
        self.max_val = 0.0
        self.test_acc = 0.0

    def _acc(self, data, number, operator):
        operators = ["+", "-", "*", "/"]
        evaluations = []

        for term in data:
            res = ""
            for i, (id, tensor) in enumerate(term.context._context.items()):
                if i % 2 == 0:
                    n = torch.argmax(
                        number.neural_model(tensor.unsqueeze(dim=0))
                    ).numpy()
                    res = res + str(n)
                else:
                    o = torch.argmax(
                        operator.neural_model(tensor.unsqueeze(dim=0))
                    ).numpy()
                    res = res + operators[o]
            try:
                res = eval(res)
                ground = term.term.arguments[0]
                evaluations.append(int(ground == res))
            except:
                evaluations.append(False)
        s = np.mean(evaluations)
        return s

    def __call__(self):
        number, operator = (
            self.store.networks["number"],
            self.store.networks["operator"],
        )
        number.neural_model.eval()
        operator.neural_model.eval()

        valid_acc = self._acc(self.valid_data, number, operator)
        if valid_acc >= self.max_val:
            self.test_acc = self._acc(self.test_data, number, operator)
            self.max_val = valid_acc

        number.neural_model.train()
        operator.neural_model.train()
        return "%s\t%s\t" % (str(valid_acc), str(self.test_acc))


def load_expression_networks(lr=1e-3, use_decoder: bool = True,
                             recon_weight: float = 1.0, emb_dim: int = 12,
                             n_mc_protos: int = 1):


    encoder = SymbolEncoder(emb_dim=emb_dim)
    decoder = SymbolDecoder(emb_dim=emb_dim) if use_decoder else None

    number_model = ProtoSymbol(
        encoder=encoder,
        n_classes=10,
        emb_dim=emb_dim,
        decoder=decoder,
        recon_weight=recon_weight,
        n_mc_protos=n_mc_protos,
    )
    operator_model = ProtoSymbol(
        encoder=encoder,                # share encoder
        n_classes=4,
        emb_dim=emb_dim,
        decoder=decoder,                # share decoder (or None)
        recon_weight=recon_weight,
        n_mc_protos=n_mc_protos,
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


def run(
    epochs=10,
    batch_size=32,
    lr=0.003,
    expression_length=None,
    expression_max_length=3,
    allow_division=True,
    device_str: str = None,
    #
    train_size=None,
    test_size=None,
    #
    log_freq=50,
    logger=print_logger,
    test_example_idx=None,
    test_batch_size=100,
    #
    seed=None,
    verbose=False,
):
    # args = epochs,batch_size,lr,expression_length,expression_max_length,allow_division,\
    #        train_size,test_size,log_freq,logger,test_example_idx,test_batch_size,seed,verbose
    #
    # logger.print("\n".join(["{}={}".format(a,b) for a in args ]) TODO
    start_time = time()
    # Setting seed for reproducibility
    set_fixed_seed(seed)

    # Load the MNIST model, and Adam optimiser
    networks = load_expression_networks(
        use_decoder=True, recon_weight=2.0, emb_dim=12, n_mc_protos=1
    )

    if device_str is not None:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the model "addition.pl" with this MNIST network
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
        file_location=str((root_path / "models/mathexpression.pl").absolute()),
        query=query,
        networks=networks,
        device=device,
    )
    optimizer = Adam(model.get_all_net_parameters(), lr=lr)
    optimizer.zero_grad()
    proving_time = time() - proving_start
    if verbose:
        logger.print("\nProving the program took {:.2f} seconds".format(proving_time))

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

    # Own DataLoader that can deal with proof trees and tensors (replicates the pytorch dataloader interface)
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=len(test_data))

    # Create test functions
    run_test_query = create_run_test_query(
        model=model,
        test_data=test_data,
        test_example_idx=test_example_idx,
        verbose=verbose,
    )
    # calculate_model_accuracy = create_model_accuracy_calculator(model, test_dataloader, start_time)
    g = GreedyDumbEvaluation(valid_data, test_data, networks)
    calculate_model_accuracy = g.header, g

    # Train the DeepStochLog model
    trainer = DeepStochLogTrainer(
        log_freq=log_freq,
        accuracy_tester=calculate_model_accuracy,
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

    logger.print("Best val accuracy:" + str(g.max_val))
    logger.print("Test accuracy:" + str(g.test_acc))

    output_dir = "output"
    print("Saving images to "+output_dir+"/" )

    save_number_prototypes(networks, out_dir=output_dir, n_samples_per_class=3)


    print("Saving models to disk")
    ckpt_file = root_path / "output" / "proto_model.pt"
    save_model_checkpoint(networks, optimizer, ckpt_file)


    return g.test_acc


if __name__ == "__main__":

    inference_only = False
    if inference_only: 
        networks, arch, opt_state = load_model_checkpoint(root_path / "output" / "proto_model.pt", device=device)

        digit_imgs, digit_labels = build_digit_bank(digits_train)
        acc_gen = generative_accuracy_numbers(networks, digit_imgs, digit_labels, use_samples=False)
        print("Generative accuracy (means):", acc_gen)

        # if you want stochastic evaluation with one sample per class:
        acc_gen_samp = generative_accuracy_numbers(networks, digit_imgs, digit_labels, use_samples=True, sample_seed=0)
        print("Generative accuracy (samples):", acc_gen_samp)

        exit()


    import os

    create_our_splits()
    logs = "logs_temp"
    if not os.path.exists(logs):
        os.mkdir(logs)
    copy2(__file__, logs)
    copy2("mathexpression_data.py", logs)
    copy2("models/mathexpression.pl", logs)

    for seed in [0, 1, 2, 3, 4]:
        folder = os.path.join(logs, "%d" % seed)
        if not os.path.exists(folder):
            os.mkdir(folder)
        for l in [3]:
            logger = PrintFileLogger(os.path.join(folder, "exact_%d.txt" % l))
            res = run(
                test_example_idx=0,
                expression_max_length=l,
                expression_length=l,
                epochs=100,
                batch_size=4,
                seed=seed,
                logger=logger,
                log_freq=100,
                allow_division=True,
                verbose=True,
                device_str="cpu"
            )
