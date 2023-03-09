:- use_module(library(lists)).
:- include_dataset(mnist_test).
:- include_network(discriminator, default(mnist_classifier)).
:- include_network(encoder, 'vae.py', 'encoder').
:- include_network(decoder, 'vae.py', 'decoder').


nn(discriminator,[Image],Digit,[0,1,2,3,4,5,6,7,8,9]) :: digit(Image,Digit) :- image(Image).

nn(encoder,[Image], Latent) :: encoder(Image, Latent).
nn(decoder,[Latent], Image) :: decoder(Latent, Image).

image(Image) :- ground(Image), encoder(Image, Latent),decoder(Latent, Image2), similar(Image, Image2).
image(Image) :- var(Image), decoder(tensor(latent(0)),Image).

similar(X,X).
P :: similar(Image1, Image2) :- Image1 \= Image2, rbf(Image1, Image2, P).

:- include_evidence('mnist.pl').