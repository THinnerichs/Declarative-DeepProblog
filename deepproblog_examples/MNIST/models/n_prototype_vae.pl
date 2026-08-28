number([],Result,Result).
number([H|T],Acc,Result) :- digit(H,Nr), Acc2 is Nr+10*Acc,number(T,Acc2,Result).
number(X,Y) :- number(X,0,Y).

multi_addition(X,Y,Z) :- number(X,X2),number(Y,Y2), Z is X2+Y2.
addition(Img1,Img2,Sum) :- digit(Img1,D1), digit(Img2,D2), Sum is D2+D1.


P0::digit(I0,0) ; P1::digit(I1,1); P2::digit(I2,2); P3::digit(I3,3); P4::digit(I4,4); P5::digit(I5,5); P6::digit(I6,6); P7::digit(I7,7); P8::digit(I8,8); P9::digit(I9,9):- all_prob([I0,I1,I2,I3,I4,I5,I6,I7,I8,I9],[0,1,2,3,4,5,6,7,8,9],[P0, P1, P2, P3, P4, P5, P6, P7, P8, P9]).

maplist(_, [], []).
maplist(P, [H1|T1], [H2|T2]) :-
    call(P, H1, H2),
    maplist(P, T1, T2).

map_encode_decode([], [], []).
map_encode_decode([Image|Images], [Prot|Prototypes], [P|Probs]) :- encode_decode(Image, Prot, P), 
map_encode_decode(Images, Prototypes, Probs).

encode_decode(Image, Prototype, P) :- encode(Image, Prototype, P1), decode(Prototype, Image, P2), mul(P1, P2, P). 

encode(Image, Prot, P) :- ground(Image), encoder(Image,Latent), lat_similar(Prot, Latent, P).
encode(Image, Prot, P) :- var(Image), sample(Prot, Sample), decoder(Sample, Image), lat_similar(Prot, Sample, P).

decode(Prot, Image, P) :- ground(Prot), sample(Prot, Latent), decoder(Latent, Image2), im_similar(Image, Image2, P).
decode(Prot, Image, 1.0) :- var(Prot), encoder(Image, Prot).

nn(encoder, [Image], Latent) :: encoder(Image, Latent).
nn(decoder, [Latent], Image) :: decoder(Latent, Image).

im_similar(X,X, 1.0).
im_similar(Image1, Image2, P) :- Image1 \= Image2, mse(Image1, Image2, P).

lat_similar(X,X, 1.0).
lat_similar(Lat1, Lat2, P) :- Lat1 \= Lat2, likelihood(Lat1, Lat2, P).

% -------------------------------------------------------------
%  PROTOTYPES: three prototypes per digit
%  prototype(C,K,tensor(prototype(C,K)))
% -------------------------------------------------------------
prototype(C, K, tensor(prototype(C,K))) :-
    between(0,9,C),
    between(0,2,K).

all_prob([], [], []).
all_prob([Image|Images], [C|Classes], [P|Probs]) :-
    class_prob(Image, C, P),
    all_prob(Images, Classes, Probs).

class_prob(Image, C, P) :-
    prototype(C, 0, Prot0), 
    encode_decode(Image, Prot0, P0),
    prototype(C, 1, Prot1), 
    encode_decode(Image, Prot1, P1),
    prototype(C, 2, Prot2), 
    encode_decode(Image, Prot2, P2),
    softmax_and_max_3(P0, P1, P2, P).


% class_prob(Image, C, P) :-
%     findall(Prot, prototype(C,_,Prot), Prots),
%     map_encode_decode_multi(Image, Prots, Scores),
%     max(Scores, P).
% 
%     % max(Scores, P). 
% 
% 
% map_encode_decode_multi(_, [], []).
% map_encode_decode_multi(Image, [Prot|Prots], [Score|Scores]) :- 
%     encode_decode(Image, Prot, Score),
%     writeln(Score),
%     map_encode_decode_multi(Image, Prots, Scores).
