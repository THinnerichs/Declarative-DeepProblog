:- use_module(library(lists)).
:- include_dataset(mnist_test).
:- include_network(discriminator, default(mnist_classifier)).
:- include_network(generator, 'mnist_generator.py').


nn(discriminator,[Image],Digit,[0,1,2,3,4,5,6,7,8,9]) :: digit(Image,Digit) :- image(Image2,Digit), similar(Image,Image2).
nn(generator, [Digit], Image) :: image(Image, Digit) :- member(Digit,[0,1,2,3,4,5,6,7,8,9]).

similar(X,X).
P :: similar(Image1, Image2) :- Image1 \= Image2, rbf(Image1, Image2, P).

:- include_evidence('mnist.pl').