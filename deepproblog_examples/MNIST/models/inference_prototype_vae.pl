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

encode(Image, Latent, P) :- ground(Image), encoder(Image,Latent2), lat_similar(Latent, Latent2, P).
encode(Image, Latent, P) :- var(Image), sample(Latent, Sample), decoder(Sample, Image), lat_similar(Latent, Sample, P).

decode(Latent, Image, 1.0) :- ground(Latent).
decode(Latent, Image, 1.0) :- var(Latent), encoder(Image, Latent).

nn(encoder, [Image], Latent) :: encoder(Image, Latent).
nn(decoder, [Latent], Image) :: decoder(Latent, Image).

lat_similar(X,X, 1.0).
lat_similar(Lat1, Lat2, P) :- Lat1 \= Lat2, likelihood(Lat1, Lat2, P).