:- use_module(library(lists)).
% :- include_dataset(mnist_test).
:- include_dataset(mnist_train).
% :- include_network(discriminator, default(mnist_classifier)).
:- include_network(encoder, 'prototype_networks.py', 'encoder').
:- include_network(decoder, 'prototype_networks.py', 'decoder').


addition(Img1,Img2,Sum) :- digit(Img1,D1), digit(Img2,D2), Sum is D2+D1.

prototype(X, tensor(prototype(X))) :- between(0,9,X). 

% nn(discriminator,[Image],Digit,[0,1,2,3,4,5,6,7,8,9]) :: digit(Image,Digit) :- image(Image, Digit).
% P :: digit(Image, Y) :- prototype(Y, Prototype), encoder(Image, Latent), rbf(Image, Prototype, P). % proposed by Robin
% P :: digit(Image, Digit) :- image(Image, Digit), prototype(Digit, Prototype), encoder(Image, Latent), cos(Latent, Prototype, P). % Tilmans solution
% digit(Image, Digit) :- image(Image), digit(Digit), prototype(Digit, Prototype), encode_decode(Image, Latent), similar(Latent, Prototype).
% digit(Image, Digit) :- prototype(Digit, Prototype), similar(Latent, Prototype), encode_decode(Image, Latent).
digit(Image, Digit) :- prototype(Digit, Prototype), encode_decode(Image, Prototype).

% encode_decode(Image, Latent) :- encode(Image, Latent1), similar(Latent, Latent1), decode(Latent, Image1), similar(Image,Image1).
encode_decode(Image, Latent) :- encode(Image, Latent), decode(Latent, Image). 


encode(Image, Latent) :- ground(Image), encoder(Image,Latent2), similar(Latent, Latent2).
encode(Image, Latent) :- var(Image), decoder(Latent, Image).

decode(Latent, Image) :- ground(Latent), decoder(Latent, Image2), similar(Image, Image2).
decode(Latent, Image) :- var(Latent), encoder(Image, Latent).

nn(encoder, [Image], Latent) :: encoder(Image, Latent).
nn(decoder, [Latent], Image) :: decoder(Latent, Image).

similar(X,X).
P :: similar(Image1, Image2) :- Image1 \= Image2, rbf(Image1, Image2, P).

:- include_evidence('mnist.pl').
