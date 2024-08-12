nn(neural1,[I1,I2,Carry],O,[0,1,2,3,4,5,6,7,8,9]) :: result(I1,I2,Carry,O).
nn(neural2,[I1,I2,Carry],NewCarry,[0,1]) :: carry(I1,I2,Carry,NewCarry).

%nn(neural1,[I],O,[0,1,2,3,4,5,6,7,8,9]) :: result(I,O).
%nn(neural2,[I],NewCarry,[0,1]) :: carry(I,NewCarry).

slot(I1,I2,Carry,Carry2,O) :-
    result(I1,I2,Carry,O),
    carry(I1,I2,Carry,Carry2).
%    one_hot(I1,10,T1),
%    one_hot(I2,10,T2),
%    one_hot(Carry,2,T3),
%    cat([T1,T2,T3],T),
%    result(T,O),
%    carry(T,Carry2).

add([],[],C,C,[]).

add([H1|T1],[H2|T2],C,Carry,[Digit|Res]) :-
    add(T1,T2,C,Carry2,Res),
    slot(H1,H2,Carry2,Carry,Digit).

add(L1,L2,C,[Carry|Res]) :- add(L1,L2,C,Carry,Res).

% Declarative formulation
prototype(X, tensor(prototype(X))) :- between(0,9,X).

result(I1,I2,Carry,O) :- prototype(O, Prototype), encode_decode(I1, I2, Carry, Prototype).

% P0::digit(I,0) ; P1::digit(I,1); P2::digit(I,2); P3::digit(I,3); P4::digit(I,4); P5::digit(I,5); P6::digit(I,6); P7::digit(I,8); P9::digit(I,9):- all_prob(I,[0,1,2,3,4,5,6,7,8,9],[P0, P1, P2, P3, P4, P5, P6, P7, P8, P9]).
P0::result(I1, I2, C, 0);P1::result(I1, I2, C, 1);P2::result(I1, I2, C, 2);P3::result(I1, I2, C, 3);P4::result(I1, I2, C, 4);P5::result(I1, I2, C, 5);P6::result(I1, I2, C, 6);P7::result(I1, I2, C, 7);P8::result(I1, I2, C, 8);P9::result(I1, I2, C, 9) :- all_prob(I1, I2, C, [0,1,2,3,4,5,6,7,8,9],[P0, P1, P2, P3, P4, P5, P6, P7, P8, P9]).

maplist(_, [], []).
maplist(P, [H1|T1], [H2|T2]) :-
    call(P, H1, H2),
    maplist(P, T1, T2).

all_prob(I1, I2, C, Classes, NormDist) :- maplist(prototype,Classes,Prototypes), encoder(I1, I2, C, Lat), likelihood_norm(Lat,Prototypes,NormDist).

encode_decode(I1, I2, C, Prototype) :- encode(I1, I2, C, Prototype), decode(Prototype, I1, I2, C).
% encode_decode(I1, I2, C, Prototype) :- encode(Image, Prototype), decode(Prototype, Image).

encode(I1, I2, C, Latent) :- ground(I1), ground(I2), ground(C), encoder(Image,Latent2), lat_similar(Latent, Latent2).
encode(I1, I2, C, Latent) :- var(Image), decoder(Latent, Image).

decode(Latent, I1, I2, C) :- ground(Latent), decoder(Latent, Image2), im_similar(Image, Image2).
decode(Latent, I1, I2, C) :- var(Latent), encoder(Image, Latent).

nn(encoder, [I1, I2, C], Latent) :: encoder(Image, Latent).
nn(decoder, [Latent], I1, I2, C) :: decoder(Latent, Image).

im_similar(X,X).
P :: im_similar(I1, I2, C1, Image2) :- Image1 \= Image2, mse(Image1, Image2, P).

% lat_similar(X,X).
%P :: lat_similar(Lat1, Lat2) :- Lat1 \= Lat2, distrcos(Lat1, Lat2, P).

lat_similar(X,X).
P :: lat_similar(Lat1, Lat2) :- Lat1 \= Lat2, likelihood(Lat1, Lat2, P).
