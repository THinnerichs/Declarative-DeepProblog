import numpy as np
from data import MapTileSource, WarcraftSP_RDD_Maps

# Try 3x3 first
src3 = MapTileSource('train', N=3, data_root='data/warcraft_shortest_path')
print("3x3: num_maps =", src3.num_maps, "first tile CHW =", src3.CHW)

ds3 = WarcraftSP_RDD_Maps('train', 'wc3_train', src3, N=3, labeled=True)
ds3.peek(5)

# Also check 12x12 tile extraction still works
src12 = MapTileSource('train', N=12, data_root='data/warcraft_shortest_path')
print("12x12: num_maps =", src12.num_maps, "first tile CHW =", src12.CHW)