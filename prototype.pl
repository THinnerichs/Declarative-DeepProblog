:- use_module(library(lists)).
:- include_dataset(mnist_test).
:- include_dataset(mnist_train).
:- include_network(discriminator, default(mnist_classifier)).
:- include_network(gen, 'prototypes.py', 'decoder').

% latent(tensor(latent(I))) :- between(1,1,I).

% addition(Img1,Img2,Sum) :- digit(Img1,D1), digit(Img2,D2), Sum is D2+D1.

prototype(X, tensor(prototype(X))) :- between(0,9,X). % creates new latents?
%prototype(0, tensor(prototype(0)).
%prototype(1, tensor(prototype(1)).
%prototype(2, tensor(prototype2)).
%prototype(3, tensor(prototype3)).
%prototype(4, tensor(prototype4)).
%prototype(5, tensor(prototype5)).
%prototype(6, tensor(prototype6)).
%prototype(7, tensor(prototype7)).
%prototype(8, tensor(prototype8)).
%prototype(9, tensor(prototype9)).


P :: digit(X, Y) :- prototype(Y, Prototype), rbf(X, Prototype, P).

nn(discriminator,[Image],Digit,[0,1,2,3,4,5,6,7,8,9]) :: digit(Image,Digit) :- image(Image, Digit).

% nn(discriminator,[Prototype], Digit, [0,1,2,3,4,5,6,7,8,9]) :: discriminator(Prototype, Digit). % Is this necessary?
% nn(gen,[Digit], Image) :: prototype(Digit, Prototype).

image(Image, Digit) :- ground(Image).
image(Image, Digit) :- var(Image), prototype(Digit, Image).%, is_prototype(Prototype).

similar(X,X).
P :: similar(Image1, Image2) :- Image1 \= Image2, rbf(Image1, Image2, P).

:- include_evidence('mnist.pl').
