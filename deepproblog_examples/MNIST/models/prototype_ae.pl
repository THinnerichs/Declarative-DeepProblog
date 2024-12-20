number([],Result,Result).
number([H|T],Acc,Result) :- digit(H,Nr), Acc2 is Nr+10*Acc,number(T,Acc2,Result).
number(X,Y) :- number(X,0,Y).

multi_addition(X,Y,Z) :- number(X,X2),number(Y,Y2), Z is X2+Y2.
addition(Img1,Img2,Sum) :- digit(Img1,D1), digit(Img2,D2), Sum is D2+D1.

prototype(X, tensor(prototype(X))) :- between(0,9,X). 

% P::digit(Image, Digit) :- prototype(Digit, Prototype), encode_decode(Image, Prototype, P).

P0::digit(I,0) ; P1::digit(I,1); P2::digit(I,2); P3::digit(I,3); P4::digit(I,4); P5::digit(I,5); P6::digit(I,6); P7::digit(I,7); P8::digit(I,8); P9::digit(I,9):- all_prob(I,[0,1,2,3,4,5,6,7,8,9],[P0, P1, P2, P3, P4, P5, P6, P7, P8, P9]).

maplist(_, [], []).
maplist(P, [H1|T1], [H2|T2]) :-
    call(P, H1, H2),
    maplist(P, T1, T2).

map_encode_decode(Image, [], []).
map_encode_decode(Image, [Prot|Prototypes], [P|Probs]) :- encode_decode(Image, Prot, P), map_encode_decode(Image, Prototypes, Probs).

all_prob(Image,Classes, Dists) :- maplist(prototype,Classes,Prototypes), maplist(encode_decode(Image), Prototypes, Dists).%, maplist(tensor_to_float, Dists, Floats), writeln(Floats).

encode_decode(Image, Prototype, P) :- encode(Image, Prototype, P1), decode(Prototype, Image, P2), mul(P1, P2, P). 

encode(Image, Latent, P) :- ground(Image), encoder(Image,Latent2), lat_similar(Latent, Latent2, P).
encode(Image, Latent, 1.0) :- var(Image), decoder(Latent, Image).

decode(Latent, Image, P) :- ground(Latent), decoder(Latent, Image2), im_similar(Image, Image2, P).%, tensor_to_float(P, Pout), writeln(Pout).
decode(Latent, Image, 1.0) :- var(Latent), encoder(Image, Latent).

nn(encoder, [Image], Latent) :: encoder(Image, Latent).
nn(decoder, [Latent], Image) :: decoder(Latent, Image).

im_similar(X,X, 1.0).
im_similar(Image1, Image2, P) :- Image1 \= Image2, mse(Image1, Image2, P).

lat_similar(X,X, 1.0).
lat_similar(Lat1, Lat2, P) :- Lat1 \= Lat2, cos(Lat1, Lat2, P).
