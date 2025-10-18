nn(mnist_net,[X],Y,[0,1,2,3,4,5,6,7,8,9]) :: digit(X,Y).

number([],Result,Result).
number([H|T],Acc,Result) :- digit(H,Nr), Acc2 is Nr+10*Acc,number(T,Acc2,Result).
number(X,Y) :- number(X,0,Y).

multi_addition(X,Y,Z) :- number(X,X2),number(Y,Y2), Z is X2+Y2.
addition(X,Y,Z) :- digit(X,X2), digit(Y,Y2), Z is X2+Y2.

% MNIST-R operators from Scallop
% 1) less_than/3 : ImageX, ImageY -> 1 if digit(X) < digit(Y) else 0
less_than(X, Y, 1) :- digit(X, DX), digit(Y, DY), DX <  DY.
less_than(X, Y, 0) :- digit(X, DX), digit(Y, DY), DX >= DY.

% 2) not_3_or_4/2 : Image -> 1 if digit ∉ {3,4} else 0
not_3_or_4(I, 1) :- digit(I, D), D \== 3, D \== 4.
not_3_or_4(I, 0) :- digit(I, 3).
not_3_or_4(I, 0) :- digit(I, 4).

% helpers
is_three(I, 1) :- digit(I, 3).
is_three(I, 0) :- digit(I, D), D \== 3.

is_3_or_4(I, 1) :- digit(I, 3).
is_3_or_4(I, 1) :- digit(I, 4).
is_3_or_4(I, 0) :- digit(I, D), D \== 3, D \== 4.

% 3) count_digit_3/2 : list of images -> number of 3s
count_digit_3([], 0).
count_digit_3([I|T], N) :-
    is_three(I, Y),
    count_digit_3(T, N0),
    N is N0 + Y.

% 4) count_3_or_4/2 : list of images -> number of 3s or 4s
count_3_or_4([], 0).
count_3_or_4([I|T], N) :-
    is_3_or_4(I, Y),
    count_3_or_4(T, N0),
    N is N0 + Y.
