dom_cost(X) :- member(X, [0,1,4]).

nn(cost_net, [X], Y, dom_cost) :: tile_cost(Y) --> [X].

map(shortest, C) -->
    tile_cost(W00),
    tile_cost(W01),
    tile_cost(W02),
    tile_cost(W10),
    tile_cost(W11),
    tile_cost(W12),
    tile_cost(W20),
    tile_cost(W21),
    tile_cost(W22),
    { shortest_3x3_rdd(W00,W01,W02,
                       W10,W11,W12,
                       W20,W21,W22,
                       C) }.

min3(A,B,C,M) :- A =< B, A =< C, M is A.
min3(A,B,C,M) :- B <  A, B =< C, M is B.
min3(A,B,C,M) :- C <  A, C <  B, M is C.

cell00(W00, C00) :-
    C00 is W00.

cell01(W01, C00, C01) :-
    C01 is C00 + W01.

cell02(W02, C01, C02) :-
    C02 is C01 + W02.

cell10(W10, C00, C10) :-
    C10 is C00 + W10.

cell20(W20, C10, C20) :-
    C20 is C10 + W20.

cell11(W11, C01, C10, C00, C11) :-
    min3(C01, C10, C00, M11),
    C11 is M11 + W11.

cell12(W12, C02, C11, C01, C12) :-
    min3(C02, C11, C01, M12),
    C12 is M12 + W12.

cell21(W21, C11, C20, C10, C21) :-
    min3(C11, C20, C10, M21),
    C21 is M21 + W21.

cell22(W22, C12, C21, C11, C22) :-
    min3(C12, C21, C11, M22),
    C22 is M22 + W22.

shortest_3x3_rdd(W00,W01,W02,
                 W10,W11,W12,
                 W20,W21,W22,
                 C) :-
    cell00(W00, C00),
    cell01(W01, C00, C01),
    cell02(W02, C01, C02),
    cell10(W10, C00, C10),
    cell20(W20, C10, C20),
    cell11(W11, C01, C10, C00, C11),
    cell12(W12, C02, C11, C01, C12),
    cell21(W21, C11, C20, C10, C21),
    cell22(W22, C12, C21, C11, C).
