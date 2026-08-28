% 3x3 grid, moves: Right, Down, Diagonal (↘). Include start & goal costs.
dom_cost(C) :- member(C,[0,1,2,3,4]).
nn(cost_net,[X],C,dom_cost) :: tile_cost(C) --> [X].

% forward search for a specific token in the remaining input
select_to(T) --> [T].
select_to(T) --> [_], select_to(T).

% top-level: only the cost is an explicit argument; tokens come from the list
sp_cost(C) --> path_cost(0,0,C).

% goal cell (2,2)
path_cost(2,2,C) -->
    select_to(tile(2,2)),
    tile_cost(C).

% step: Right
path_cost(I,J,C) -->
    { I<2, I1 is I+1, J1 is J },
    select_to(tile(I,J)),
    tile_cost(Cij),
    path_cost(I1,J1,R),
    { C is Cij + R }.

% step: Down
path_cost(I,J,C) -->
    { J<2, I1 is I, J1 is J+1 },
    select_to(tile(I,J)),
    tile_cost(Cij),
    path_cost(I1,J1,R),
    { C is Cij + R }.

% step: Diagonal
path_cost(I,J,C) -->
    { I<2, J<2, I1 is I+1, J1 is J+1 },
    select_to(tile(I,J)),
    tile_cost(Cij),
    path_cost(I1,J1,R),
    { C is Cij + R }.
