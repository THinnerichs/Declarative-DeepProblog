#!/usr/bin/env python3
"""
End-to-end DeepProbLog training on map-level shortest path (Right/Down/Diagonal),
restricted to N=3 for tractability.

Query: sp_cost(map(M), Y_gold)
Program: models/warcraft_sp_monotone.pl (injects grid facts for N=3)
Neural predicate: nn(cost_net, [tile(ID)], Cost) with 5 classes (0..4)
"""

import os
import pickle
import argparse
import numpy as np
import torch

from sklearn.metrics import accuracy_score, mean_absolute_error

from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.engines import ExactEngine
from deepproblog.logger import VerboseLogger

from data import MapTileSource, WarcraftSP_RDD_Maps
from networks.network import TileCostCNN


def program_with_grid(N: int) -> str:
    assert N == 3, "This end-to-end program is intended for N=3."
    header = [f"grid_n({N})."]
    for i in range(N):
        for j in range(N):
            header.append(f"tile_pos({i},{j}).")
    header.append("")  # blank line
    with open("models/warcraft_sp_monotone.pl", "r") as f:
        body = f.read()
    return "\n".join(header) + "\n" + body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=3, choices=[3], help="Grid size (fixed to 3 for tractability).")
    ap.add_argument("--data_root", type=str, default="data/warcraft_shortest_path")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--state", type=str, default="saved_models/warcraft_sp.pkl")
    args = ap.parse_args()

    # Tensor sources
    train_src = MapTileSource("train", N=args.N, data_root=args.data_root)
    val_src   = MapTileSource("val",   N=args.N, data_root=args.data_root)
    test_src  = MapTileSource("test",  N=args.N, data_root=args.data_root)

    # Map-level datasets (labels are R/D/Diag shortest-path totals)
    train_set = WarcraftSP_RDD_Maps("train", f"wc{args.N}_train", train_src, N=args.N, labeled=True)
    val_set_L = WarcraftSP_RDD_Maps("val",   f"wc{args.N}_valL",  val_src,   N=args.N, labeled=True)
    val_set_Q = WarcraftSP_RDD_Maps("val",   f"wc{args.N}_valQ",  val_src,   N=args.N, labeled=False)
    test_set_Q= WarcraftSP_RDD_Maps("test",  f"wc{args.N}_testQ", test_src,  N=args.N, labeled=False)

    # CNN -> DPL network
    in_ch = train_src.CHW[0]
    net = TileCostCNN(in_ch=in_ch, num_classes=5)
    dpl_net = Network(net, "cost_net")
    dpl_net.optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    # Full program (grid facts + logic)
    program_string = program_with_grid(args.N)

    logger = VerboseLogger(log_every=50)
    model = Model(program_string, [dpl_net], logger=logger)
    engine = ExactEngine(model, cache_memory=True)

    # Register tensor sources (names must match dataset source_name when used)
    model.add_tensor_source(f"wc{args.N}_train", train_src)
    model.add_tensor_source(f"wc{args.N}_valL",  val_src)
    model.add_tensor_source(f"wc{args.N}_valQ",  val_src)
    model.add_tensor_source(f"wc{args.N}_testQ", test_src)

    # Train / restore
    if os.path.isfile(args.state):
        with open(args.state, "rb") as f:
            model.__setstate__(pickle.load(f))
        print(f"Restored model state from {args.state}")
    else:
        print("Training end-to-end on sp_cost (N=3, R/D/Diag)…")
        model.fit(dataset=train_set, engine=engine, batch_size=args.batch, shuffle=True, stop_condition=args.epochs)
        os.makedirs(os.path.dirname(args.state), exist_ok=True)
        with open(args.state, "wb") as f:
            pickle.dump(model.__getstate__(), f)

    # Validate: predict Y for sp_cost(map(M), Y)
    y_val_hat = np.array(model.predict(dataset=val_set_Q, engine=engine), dtype=np.int64)
    y_val_true= val_set_L.gold_labels()
    print(f"[VAL] exact-match accuracy: {accuracy_score(y_val_true, y_val_hat):.4f}")
    print(f"[VAL] MAE: {mean_absolute_error(y_val_true, y_val_hat):.3f}")
    print(f"[VAL] first 10 (gold, pred): {list(zip(y_val_true[:10].tolist(), y_val_hat[:10].tolist()))}")

    # Test (unlabeled)
    y_test_hat = np.array(model.predict(dataset=test_set_Q, engine=engine), dtype=np.int64)
    print(f"[TEST] first 10 predictions: {y_test_hat[:10].tolist()}")


if __name__ == "__main__":
    main()
