number([],Result,Result).
number([H|T],Acc,Result) :- digit(H,Nr), Acc2 is Nr+10*Acc,number(T,Acc2,Result).
number(X,Y) :- number(X,0,Y).

multi_addition(X,Y,Z) :- number(X,X2),number(Y,Y2), Z is X2+Y2.
addition(Img1,Img2,Sum) :- digit(Img1,D1), digit(Img2,D2), Sum is D2+D1.

prototype(X, tensor(prototype(X))) :- between(0,9,X). 

P0::digit(I0,0) ; P1::digit(I1,1); P2::digit(I2,2); P3::digit(I3,3); P4::digit(I4,4); P5::digit(I5,5); P6::digit(I6,6); P7::digit(I7,7); P8::digit(I8,8); P9::digit(I9,9):- all_prob([I0,I1,I2,I3,I4,I5,I6,I7,I8,I9],[0,1,2,3,4,5,6,7,8,9],[P0, P1, P2, P3, P4, P5, P6, P7, P8, P9]).

maplist(_, [], []).
maplist(P, [H1|T1], [H2|T2]) :-
    call(P, H1, H2),
    maplist(P, T1, T2).

map_encode_decode([], [], []).
map_encode_decode([Image|Images], [Prot|Prototypes], [P|Probs]) :- encode_decode(Image, Prot, P), map_encode_decode(Images, Prototypes, Probs).

all_prob(Images,Classes, Dists) :- maplist(prototype,Classes,Prototypes), map_encode_decode(Images, Prototypes, Dists).

encode_decode(Image, Prototype, P) :- encode(Image, Prototype, P1), decode(Prototype, Image, P2), mul(P1, P2, P). 

encode(Image, Prot, P) :- ground(Image), encoder(Image,Latent), lat_similar(Prot, Latent, P).
encode(Image, Prot, P) :- var(Image), sample(Prot, Sample), decoder(Sample, Image), lat_similar(Prot, Sample, P).

decode(Prot, Image, P) :- ground(Prot), sample(Prot, Latent), decoder(Latent, Image2), im_similar(Image, Image2, P).
decode(Prot, Image, 1.0) :- var(Prot), encoder(Image, Prot).

nn(encoder, [Image], Latent) :: encoder(Image, Latent).
nn(decoder, [Latent], Image) :: decoder(Latent, Image).

im_similar(X,X, 1.0).
im_similar(Image1, Image2, P) :- Image1 \= Image2, mse(Image1, Image2, P).

lat_similar(X,X, 1.0).
lat_similar(Lat1, Lat2, P) :- Lat1 \= Lat2, likelihood(Lat1, Lat2, P).


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

% sum_list/2: sum of digit values in a list of images
sum_list([], 0).
sum_list([I|T], S) :-
    digit(I, D),
    sum_list(T, S0),
    S is S0 + D.

% Datasets will always provide lists of the right length.
% We keep separate predicate names for clarity and metrics.
sum2(Imgs, S) :- sum_list(Imgs, S).  % expects [I1, I2]
sum3(Imgs, S) :- sum_list(Imgs, S).  % expects [I1, I2, I3]
sum4(Imgs, S) :- sum_list(Imgs, S).  % expects [I1, I2, I3, I4]


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
