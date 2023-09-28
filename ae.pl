:- use_module(library(lists)).
:- include_dataset(mnist_test).
:- include_dataset(mnist_train).
:- include_network(discriminator, default(mnist_classifier)).
:- include_network(encoder, 'vae.py', 'encoder').
:- include_network(decoder, 'vae.py', 'decoder').

latent(tensor(latent(I))) :- between(1,1,I).

addition(Img1,Img2,Sum) :- digit(Img1,D1), digit(Img2,D2), Sum is D2+D1.

nn(discriminator,[Image],Digit,[0,1,2,3,4,5,6,7,8,9]) :: digit(Image,Digit) :- image(Image).

nn(encoder,[Image], Latent) :: encoder(Image, Latent).
nn(decoder,[Latent], Image) :: decoder(Latent, Image).

image(Image) :- ground(Image), encoder(Image, Latent),decoder(Latent, Image2), similar(Image, Image2).
image(Image) :- var(Image),latent(Latent), decoder(Latent, Image).

similar(X,X).
P :: similar(Image1, Image2) :- Image1 \= Image2, rbf(Image1, Image2, P).

:- include_evidence('mnist.pl').