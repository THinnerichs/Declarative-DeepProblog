from pathlib import Path
from time import time

import torch
from torch.optim import Adam

from deepstochlog.model import DeepStochLogModel
from deepstochlog.dataloader import DataLoader
from deepstochlog.trainer import DeepStochLogTrainer, print_logger
from deepstochlog.term import Term, List
from deepstochlog.utils import (
    set_fixed_seed,
    create_model_accuracy_calculator,
    create_run_test_query,
)

from data.warcraft_dataset_ds import WarcraftSP3Dataset_DS
from networks.network import build_networks

from utils import WarcraftTileAccuracy, make_subset


root_path = Path(__file__).parent


def create_map_query(total_cost="_"):
    total_cost_term = Term(str(total_cost))
    tiles_arg = List(*[Term(f"tile_{i}_{j}") for i in range(3) for j in range(3)])
    return Term("map", Term("shortest"), total_cost_term, tiles_arg)


def main(
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 1e-3,
    seed: int | None = 0,
    device_str: str | None = None,
):
    if seed is not None:
        set_fixed_seed(seed)

    # --- full datasets ---
    full_train_ds = WarcraftSP3Dataset_DS(split="train")
    full_val_ds   = WarcraftSP3Dataset_DS(split="val")
    full_test_ds  = WarcraftSP3Dataset_DS(split="test")

    # choose how many you want
    n_train = 500
    n_val   = 100
    n_test  = 100

    # --- create random subsets ---
    train_ds, train_idx = make_subset(full_train_ds, n_train, seed if seed is not None else 0)
    val_ds,   val_idx   = make_subset(full_val_ds,   n_val,   (seed or 0) + 1)
    test_ds,  test_idx  = make_subset(full_test_ds,  n_test,  (seed or 0) + 2)

    # --- dataloaders on subsets ---
    train_loader = DataLoader(train_ds, batch_size=batch_size)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader  = DataLoader(test_ds, batch_size=batch_size)


    print("Train samples:\t",len(train_ds))
    print("Valid samples:\t",len(val_ds))
    print("Test samples:\t",len(test_ds))

    # --- Channels / networks ---
    in_ch = full_train_ds.src.CHW[0]
    print("in_ch =", in_ch)

    networks = build_networks(in_ch)

    # --- Device ---
    if device_str is not None:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # --- Program + proto query (like HWF) ---
    query = [create_map_query("_")]

    model = DeepStochLogModel.from_file(
        file_location=str((root_path / "models/warcraft_sp.pl").absolute()),
        query=query,
        networks=networks,
        device=device,
    )

    optimizer = Adam(model.get_all_net_parameters(), lr=lr)

    # --- Accuracy tester + example query for logging ---
    start_time = time()
    tile_acc = WarcraftTileAccuracy(
        src=full_test_ds.src,   # MapTileSource with all test maps
        indices=test_idx,       # only evaluate on the chosen subset
        networks=networks,
    )
    accuracy_tester = (tile_acc.header, tile_acc)

    run_test_query = create_run_test_query(
        model=model,
        test_data=val_loader,
        test_example_idx=0,
        verbose=False,
    )

    # --- Trainer ---
    trainer = DeepStochLogTrainer(
        log_freq=1000,
        accuracy_tester=accuracy_tester,  # this is a (header, fn) pair
        logger=print_logger,
        print_time=False,
        test_query=run_test_query,
    )

    trainer.train(
        model=model,
        optimizer=optimizer,
        dataloader=train_loader,
        epochs=epochs,
    )

    print("Training complete.")


if __name__ == "__main__":
    main()
