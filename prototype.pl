:- use_module(library(lists)).
:- include_dataset(mnist_test).
:- include_dataset(mnist_train).
:- include_network(discriminator, default(mnist_classifier)).
:- include_network(gen, 'prototypes.py', 'decoder').

latent(tensor(latent(I))) :- between(1,1,I).

addition(Img1,Img2,Sum) :- digit(Img1,D1), digit(Img2,D2), Sum is D2+D1.

prototype(0, Prototype0).
prototype(1, Prototype1).
prototype(2, Prototype2).
prototype(3, Prototype3).
prototype(4, Prototype4).
prototype(5, Prototype5).
prototype(6, Prototype6).
prototype(7, Prototype7).
prototype(8, Prototype8).
prototype(9, Prototype9).

nn(discriminator,[Image],Digit,[0,1,2,3,4,5,6,7,8,9]) :: digit(Image,Digit) :- image(Image).

% nn(discriminator,[Image], Latent) :: discriminator(Image, Latent).
nn(gen,[Digit], Image) :: prototype(Digit, Image).

image(Image) :- ground(Image).
P :: image(Image) :- var(Image), prototype(Digit, Image), discriminator(Image, Digit, P).

similar(X,X).
P :: similar(Image1, Image2) :- Image1 \= Image2, rbf(Image1, Image2, P).

:- include_evidence('mnist.pl').
