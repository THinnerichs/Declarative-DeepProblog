import numpy as np
import torch
from typing import Dict

from deepstochlog.dataset import ContextualizedTermDataset, ContextualizedTerm
from deepstochlog.context import Context
from deepstochlog.term import Term, List

from data import (
    _extract_tile_from_map,
    _normalize_tile,
    MapTileSource,
    rdd_shortest_cost,
)


class WarcraftSP3Dataset_DS(ContextualizedTermDataset):
    """
    DeepStochlog dataset for 3x3 Warcraft shortest-path.

    Each item is a ContextualizedTerm:

        map(shortest, Cost, [tile_0_0, tile_0_1, ..., tile_2_2])

    where each tile_* term is mapped in the Context to its image tensor.
    """

    def __init__(
        self,
        split: str,
        data_root: str = "data/warcraft_shortest_path/3x3_rebinned",
        N: int = 3,
    ):
        super().__init__()
        self.split = split
        self.N = N

        # Use your existing MapTileSource loader
        self.src = MapTileSource(split=split, N=N, data_root=data_root)
        self.maps = self.src.maps          # array of maps
        self.weights = self.src.vw         # vertex weights
        self.num_maps = len(self.maps)

        # Precompute ground-truth shortest-path costs
        self.labels = [rdd_shortest_cost(np.array(w)) for w in self.weights]

        # Reuse the same Term objects for all tiles across all examples
        self.tile_terms = [
            Term(f"tile_{i}_{j}") for i in range(N) for j in range(N)
        ]

    def __len__(self):
        return self.num_maps

    def __getitem__(self, item):
        # IMPORTANT: handle slices, because DataLoader does self.dataset[:len]
        if isinstance(item, slice):
            return (self[i] for i in range(*item.indices(len(self))))

        idx = int(item)
        N = self.N

        context_dict: Dict[Term, torch.Tensor] = {}

        # Build context: term -> tensor
        for i in range(N):
            for j in range(N):
                # Single map: shape like (H, W, C) or (C, H, W)
                tile_np = _extract_tile_from_map(self.maps[idx], N, i, j)
                tile_np = _normalize_tile(tile_np)
                tile_tensor = torch.from_numpy(tile_np)

                term_key = self.tile_terms[i * N + j]
                context_dict[term_key] = tile_tensor

        # List of tile terms in a fixed order
        tile_list = List(*self.tile_terms)

        # Ground-truth shortest-path cost for this map
        cost_term = Term(str(int(self.labels[idx])))

        # Logical term: map(shortest, Cost, TileList)
        term = Term("map", Term("shortest"), cost_term, tile_list)

        return ContextualizedTerm(
            context=Context(context_dict),
            term=term,
        )
