% :- use_module(library(lists)).
% :- include_dataset(mnist_test).
% :- include_dataset(mnist_train).
% :- include_network(discriminator, default(mnist_classifier)).
% :- include_network(encoder, 'prototype_networks.py', 'encoder').
% :- include_network(decoder, 'prototype_networks.py', 'decoder').

number([],Result,Result).
number([H|T],Acc,Result) :- digit(H,Nr), Acc2 is Nr+10*Acc,number(T,Acc2,Result).
number(X,Y) :- number(X,0,Y).

multi_addition(X,Y,Z) :- number(X,X2),number(Y,Y2), Z is X2+Y2.
addition(Img1,Img2,Sum) :- digit(Img1,D1), digit(Img2,D2), Sum is D2+D1.

prototype(X, tensor(prototype(X))) :- between(0,9,X). 

digit(Image, Digit) :- prototype(Digit, Prototype), encode_decode(Image, Prototype).

encode_decode(Image, Latent) :- encode(Image, Latent), decode(Latent, Image). 

encode(Image, Latent) :- ground(Image), encoder(Image,Latent2), similar(Latent, Latent2).
encode(Image, Latent) :- var(Image), decoder(Latent, Image).

decode(Latent, Image) :- ground(Latent), decoder(Latent, Image2), similar(Image, Image2).
decode(Latent, Image) :- var(Latent), encoder(Image, Latent).

nn(encoder, [Image], Latent) :: encoder(Image, Latent).
nn(decoder, [Latent], Image) :: decoder(Latent, Image).

similar(X,X).
P :: similar(Image1, Image2) :- Image1 \= Image2, rbf(Image1, Image2, P).

% :- include_evidence('mnist.pl').
