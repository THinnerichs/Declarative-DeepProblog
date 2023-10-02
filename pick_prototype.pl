:- use_module(library(lists)).
:- include_dataset(mnist_test).
:- include_dataset(mnist_train).
:- include_network(discriminator, default(mnist_classifier)).

addition(Img1,Img2,Sum) :- digit(Img1,D1), digit(Img2,D2), Sum is D2+D1.

prototype(X, tensor(prototype(X))) :- between(0,9,X).

nn(discriminator,[Image],Digit,[0,1,2,3,4,5,6,7,8,9]) :: digit(Image,Digit) :- image(Image, Digit).

image(Image, Digit) :- ground(Image).
image(Image, Digit) :- var(Image), prototype(Digit, Image).

:- include_evidence('mnist.pl').
