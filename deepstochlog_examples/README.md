# DeepStochLog

Needs `deepstochlog` on the path (see the root README).

## HWF (`mathexpression/`)

```bash
./mathexpression/download_hwf.sh                    # data into data/raw
python mathexpression/mathexpression.py             # vanilla DeepStochLog
python mathexpression/mathexpression_prototype.py   # declarative (ProtoVAE)
python mathexpression/mathexpression_prototype.py --inference-only --calc-gen-acc
```

`--save-proto-images` writes the decoded prototypes; `--checkpoint` sets where
the model is stored.

## WarCraft pathfinding (`WarcraftPathfinding/`)

```bash
./WarcraftPathfinding/download.sh                   # 12x12 maps
python WarcraftPathfinding/data/make_3x3_subset.py  # 3x3 subset used in the paper
python WarcraftPathfinding/run_warcraft_pathfinding.py       # vanilla DeepStochLog
python WarcraftPathfinding/warcraft_vae.py --N 3 --gen_cost 12 --gen_num 5
```

`--gen_cost` generates maps whose shortest path has that cost.
