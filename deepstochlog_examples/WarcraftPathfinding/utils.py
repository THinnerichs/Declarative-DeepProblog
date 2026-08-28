from itertools import product

def shortest_path_cost_for_grid(grid, n):
    """
    grid: flat list of length n*n with cell costs
    n: size of the grid (n x n)
    """
    # Convert flat list to 2D list for convenience
    cost = [grid[i*n:(i+1)*n] for i in range(n)]

    # dp[i][j] = minimum cost to reach cell (i, j) from (0, 0)
    dp = [[float('inf')] * n for _ in range(n)]
    dp[0][0] = cost[0][0]

    for i in range(n):
        for j in range(n):
            if i == 0 and j == 0:
                continue  # already set
            candidates = []
            if i > 0:
                candidates.append(dp[i-1][j])       # from above
            if j > 0:
                candidates.append(dp[i][j-1])       # from left
            if i > 0 and j > 0:
                candidates.append(dp[i-1][j-1])     # from diagonal up-left
            dp[i][j] = cost[i][j] + min(candidates)

    return dp[n-1][n-1]


def all_possible_shortest_path_costs(cost_values, n):
    """
    cost_values: list of possible costs for each cell (e.g., [0, 1, 4])
    n: grid size (n x n)

    Returns a sorted list of all distinct shortest-path costs
    over all possible n x n grids with cell values from cost_values.
    """
    unique_costs = set()

    # All possible assignments of costs to n*n cells
    for grid in product(cost_values, repeat=n*n):
        min_cost = shortest_path_cost_for_grid(grid, n)
        unique_costs.add(min_cost)

    return sorted(unique_costs)

def make_subset(dataset, n_samples, seed=0):
    """
    Return (subset_list, indices) where subset_list is a list of ContextualizedTerm,
    and indices are the original positions in the full dataset.
    """
    rng = np.random.RandomState(seed)
    n = min(n_samples, len(dataset))
    idx = rng.choice(len(dataset), size=n, replace=False)
    subset = [dataset[i] for i in idx]
    return subset, idx

import torch
import numpy as np

COST_TO_CLASS = {0: 0, 1: 1, 4: 2}  # assuming rebinned to {0,1,4}

class WarcraftTileAccuracy:
    def __init__(self, src, indices, networks, cost_to_class=None):
        """
        src         : MapTileSource (e.g. full_test_ds.src)
        indices     : 1D array/list of map indices to evaluate on (subset)
        networks    : NetworkStore (with networks["cost_net"])
        cost_to_class : dict mapping rebinned cost -> class index (default {0,1,4}->{0,1,2})
        """
        self.src = src
        self.indices = np.array(indices, dtype=int)
        self.store = networks
        self.header = "TileAcc\t"
        self.cost_to_class = cost_to_class or COST_TO_CLASS

    def __call__(self):
        net = self.store.networks["cost_net"]
        model = net.neural_model

        model.eval()
        correct = 0
        total = 0

        N = self.src.N  # grid size, e.g. 3

        for m in self.indices:
            for i in range(N):
                for j in range(N):
                    # global id as in your MapTileSource __call__
                    gid = m * (N * N) + i * N + j
                    x = self.src(gid).unsqueeze(0)  # (1, C, H, W)

                    with torch.no_grad():
                        logits = model(x)
                        pred_class = int(torch.argmax(logits, dim=1))

                    true_cost = int(np.array(self.src.vw[m])[i, j])
                    true_class = self.cost_to_class[true_cost]

                    if pred_class == true_class:
                        correct += 1
                    total += 1

        acc = correct / total if total > 0 else 0.0
        model.train()
        return f"{acc:.4f}\t"


if __name__ == "__main__":
    # Example usage:
    cost_values = [0, 1, 4]
    n = 3

    result = all_possible_shortest_path_costs(cost_values, n)
    print(f"All possible shortest-path costs for cost_values={cost_values}, n={n}:")
    print(result)