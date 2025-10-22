% models/warcraft_sp_rdd.pl
% End-to-end map-level shortest path with Right, Down, and Diagonal (↘︎) moves.
% The runner injects grid_n(N). and tile_pos(I,J). facts for N=3.

% Neural predicate: cost class in {0..4} for a tile ID.
nn(cost_net, [Tile], Cost, [0.0,1.0,2.0,3.0,4.0]) :: tile_cost(Tile, Cost).

% Map tile ID from (map index, I, J):
% ID = Map * (N*N) + I*N + J
tile_id(map(M), I, J, ID) :-          % handle wrapped form map(M)
    grid_n(N),
    ID is M * (N*N) + I * N + J.

tile_id(M, I, J, ID) :-               % also accept plain integer M
    grid_n(N),
    ID is M * (N*N) + I * N + J.


% Node cost delegates to the neural predicate
node_cost(Map, I, J, C) :-
    tile_id(Map, I, J, ID),
    tile_cost(tile(ID), C).

% --- Dynamic programming over monotone moves (R, D, Diag ↘︎) ---

% Base:
rdd_cost(Map, 0, 0, C) :- node_cost(Map, 0, 0, C).

% First row: can only come from left (Right moves)
rdd_cost(Map, 0, J, C) :-
    J > 0, J1 is J - 1,
    rdd_cost(Map, 0, J1, C1),
    node_cost(Map, 0, J, Cij),
    C is C1 + Cij.

% First column: can only come from above (Down moves)
rdd_cost(Map, I, 0, C) :-
    I > 0, I1 is I - 1,
    rdd_cost(Map, I1, 0, C1),
    node_cost(Map, I, 0, Cij),
    C is C1 + Cij.

% Interior: three predecessors (Up, Left, Diagonal)
rdd_cost(Map, I, J, C) :-
    I > 0, J > 0,
    I1 is I - 1, J1 is J - 1,
    (
        rdd_cost(Map, I1, J,  C1)      % from above
    ;   rdd_cost(Map, I,  J1, C1)      % from left
    ;   rdd_cost(Map, I1, J1, C1)      % from diagonal ↘︎
    ),
    node_cost(Map, I, J, Cij),
    C is C1 + Cij.

% List minimum
minimum([X], X).
minimum([H|T], M) :- minimum(T, M1), (H < M1 -> M = H ; M = M1).

% Shortest path cost from (0,0) to (N-1,N-1)
sp_cost(Map, Total) :-
    grid_n(N),
    I is N - 1, J is N - 1,
    findall(C, rdd_cost(Map, I, J, C), Cs),
    minimum(Cs, Total).
