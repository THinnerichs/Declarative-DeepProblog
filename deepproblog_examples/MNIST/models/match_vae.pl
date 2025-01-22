number([],Result,Result).
number([H|T],Acc,Result) :- digit(H,Nr), Acc2 is Nr+10*Acc,number(T,Acc2,Result).
number(X,Y) :- number(X,0,Y).

multi_addition(X,Y,Z) :- number(X,X2),number(Y,Y2), Z is X2+Y2.
addition(Img1,Img2,Sum) :- digit(Img1,D1), digit(Img2,D2), Sum is D2+D1.

prototype(X, tensor(prototype(X))) :- between(0,9,X). 

P0::digit(I,0) ; P1::digit(I,1); P2::digit(I,2); P3::digit(I,3); P4::digit(I,4); P5::digit(I,5); P6::digit(I,6); P7::digit(I,7); P8::digit(I,8); P9::digit(I,9):- 
    prototype(0, Prot0), 
    prototype(1, Prot1), 
    prototype(2, Prot2), 
    prototype(3, Prot3), 
    prototype(4, Prot4), 
    prototype(5, Prot5), 
    prototype(6, Prot6), 
    prototype(7, Prot7), 
    prototype(8, Prot8), 
    prototype(9, Prot9),
    prototype_match(I, Prot0, P0),
    prototype_match(I, Prot1, P1),
    prototype_match(I, Prot2, P2),
    prototype_match(I, Prot3, P3),          
    prototype_match(I, Prot4, P4),
    prototype_match(I, Prot5, P5),
    prototype_match(I, Prot6, P6),       
    prototype_match(I, Prot7, P7),
    prototype_match(I, Prot8, P8),
    prototype_match(I, Prot9, P9).

prototype_match(Image, Prot, P) :- encode(Image, Prot, P1), decode(Prot, Image, P2), mul(P1, P2, P).

encode(Image, Latent, P) :- ground(Image), encoder(Image,Latent2), lat_similar(Latent, Latent2, P).
encode(Image, Latent, 1.0) :- var(Image), decoder(Latent, Image).

decode(Latent, Image, P) :- ground(Latent), decoder(Latent, Image2), im_similar(Image, Image2, P).
decode(Latent, Image, 1.0) :- var(Latent), encoder(Image, Latent).

nn(encoder, [Image], Latent) :: encoder(Image, Latent).
nn(decoder, [Latent], Image) :: decoder(Latent, Image).

im_similar(X,X, 1.0).
im_similar(Image1, Image2, P) :- Image1 \= Image2, mse(Image1, Image2, P).

lat_similar(X,X, 1.0).
lat_similar(Lat1, Lat2, P) :- Lat1 \= Lat2, likelihood(Lat1, Lat2, P).
