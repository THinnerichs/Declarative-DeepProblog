% models/warcraft_costs.pl
% Prototype DeepProbLog program for Warcraft 12x12 tile-cost classification.
% The neural predicate maps a 3x8x8 tile image to a discrete cost in {0,1,2,3,4}.

% Neural classifier: tile image -> cost class
nn(cost_net, [Tile], Cost) :: tile_cost(Tile, Cost).

% (Optional) keep costs in range 0..4 — helpful in debugging, not strictly needed:
valid_cost(0). valid_cost(1). valid_cost(2). valid_cost(3). valid_cost(4).
tile_cost_checked(Tile, Cost) :- tile_cost(Tile, Cost), valid_cost(Cost).
