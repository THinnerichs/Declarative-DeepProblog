:- use_module(library(lists)).
:- include_dataset(mnist_test).
:- include_dataset(mnist_train).
:- include_network(discriminator, default(mnist_classifier)).

% latent(tensor(latent(I))) :- between(1,1,I).

addition(Img1,Img2,Sum) :- digit(Img1,D1), digit(Img2,D2), Sum is D2+D1.

prototype(X, tensor(prototype(X))) :- between(0,9,X). % creates new latents?

nn(discriminator,[Image],Digit,[0,1,2,3,4,5,6,7,8,9]) :: digit(Image,Digit) :- image(Image, Digit).

image(Image, Digit) :- ground(Image).
image(Image, Digit) :- var(Image), prototype(Digit, Image).%, is_prototype(Prototype).

% P :: digit(X, Y) :- prototype(Y, Prototype), rbf(X, Prototype, P).
% similar(X,X).
% P :: similar(Image1, Image2) :- Image1 \= Image2, rbf(Image1, Image2, P).

:- include_evidence('mnist.pl').