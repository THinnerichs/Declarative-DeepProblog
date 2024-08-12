number([],Result,Result).
number([H|T],Acc,Result) :- digit(H,Nr), Acc2 is Nr+10*Acc,number(T,Acc2,Result).
number(X,Y) :- number(X,0,Y).

multi_addition(X,Y,Z) :- number(X,X2),number(Y,Y2), Z is X2+Y2.
addition(Img1,Img2,Sum) :- digit(Img1,D1), digit(Img2,D2), Sum is D2+D1.

prototype(X, tensor(prototype(X))) :- between(0,9,X). 

digit(Image, Digit) :- prototype(Digit, Prototype), encode_decode(Image, Prototype).

P0::digit(I,0) ; P1::digit(I,1); P2::digit(I,2); P3::digit(I,3); P4::digit(I,4); P5::digit(I,5); P6::digit(I,6); P7::digit(I,8); P9::digit(I,9):- all_prob(I,[0,1,2,3,4,5,6,7,8,9],[P0, P1, P2, P3, P4, P5, P6, P7, P8, P9]).

maplist(_, [], []).
maplist(P, [H1|T1], [H2|T2]) :-
    call(P, H1, H2),
    maplist(P, T1, T2).

all_prob(Image,Classes, NormDist) :- maplist(prototype,Classes,Prototypes), encoder(Image,Lat), likelihood_norm(Lat,Prototypes,NormDist).

encode_decode(Image, Prototype) :- encode(Image, Prototype), decode(Prototype, Image). 

encode(Image, Latent) :- ground(Image), encoder(Image,Latent2), lat_similar(Latent, Latent2).
encode(Image, Latent) :- var(Image), decoder(Latent, Image).

decode(Latent, Image) :- ground(Latent), decoder(Latent, Image2), im_similar(Image, Image2).
decode(Latent, Image) :- var(Latent), encoder(Image, Latent).

nn(encoder, [Image], Latent) :: encoder(Image, Latent).
nn(decoder, [Latent], Image) :: decoder(Latent, Image).

im_similar(X,X).
P :: im_similar(Image1, Image2) :- Image1 \= Image2, mse(Image1, Image2, P).

lat_similar(X,X).
P :: lat_similar(Lat1, Lat2) :- Lat1 \= Lat2, likelihood(Lat1, Lat2, P).

