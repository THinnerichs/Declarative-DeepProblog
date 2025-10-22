% Core arithmetic (unchanged)
number([],Result,Result).
number([H|T],Acc,Result) :- digit(H,Nr), Acc2 is Nr+10*Acc, number(T,Acc2,Result).
number(X,Y) :- number(X,0,Y).

multi_addition(X,Y,Z) :- number(X,X2), number(Y,Y2), Z is X2+Y2.
addition(Img1,Img2,Sum) :- digit(Img1,D1), digit(Img2,D2), Sum is D2+D1.

% Prototypes: binary index (C,K)
prototype(C, K, tensor(prototype(C,K))) :-
    between(0,9,C),
    between(0,2,K).

% Build the 3 prototypes for a single class C using maplist + call (closure prototype(C))
% Ks = [0,1,2]  -->  Prots = [tensor(prototype(C,0)), tensor(prototype(C,1)), tensor(prototype(C,2))]
class_prototypes(C, Prots) :-
    Ks = [0,1,2],
    maplist(prototype(C), Ks, Prots).

% Build Prototypes as a list of lists for all classes in Classes
% Classes = [0,1,...,9]  -->  Prototypes = [[...3 terms for 0...], [...3 terms for 1...], ...]
classes_prototypes(Classes, Prototypes) :-
    maplist(class_prototypes, Classes, Prototypes).

% AD for digit classification (unchanged interface to VAE)
P0::digit(I0,0) ; P1::digit(I1,1) ; P2::digit(I2,2) ; P3::digit(I3,3) ; P4::digit(I4,4) ;
P5::digit(I5,5) ; P6::digit(I6,6) ; P7::digit(I7,7) ; P8::digit(I8,8) ; P9::digit(I9,9) :-
    all_prob([I0,I1,I2,I3,I4,I5,I6,I7,I8,I9],
             [0,1,2,3,4,5,6,7,8,9],
             [P0,P1,P2,P3,P4,P5,P6,P7,P8,P9]).

% Simple maplist/3
maplist(_, [], []).
maplist(P, [H1|T1], [H2|T2]) :-
    call(P, H1, H2),
    maplist(P, T1, T2).

% Encode-decode and helpers (unchanged core)
encode_decode(Image, Prototype, P) :-
    encode(Image, Prototype, P1),
    decode(Prototype, Image, P2),
    mul(P1, P2, P).

encode(Image, Prot, P) :-
    ground(Image),
    encoder(Image,Latent),
    lat_similar(Prot, Latent, P).
encode(Image, Prot, P) :-
    var(Image),
    sample(Prot, Sample),
    decoder(Sample, Image),
    lat_similar(Prot, Sample, P).

decode(Prot, Image, P) :-
    ground(Prot),
    sample(Prot, Latent),
    decoder(Latent, Image2),
    im_similar(Image, Image2, P).
decode(Prot, Image, 1.0) :-
    var(Prot),
    encoder(Image, Prot).

nn(encoder, [Image], Latent) :: encoder(Image, Latent).
nn(decoder, [Latent], Image) :: decoder(Latent, Image).

im_similar(X,X, 1.0).
im_similar(Image1, Image2, P) :- Image1 \= Image2, mse(Image1, Image2, P).

lat_similar(X,X, 1.0).
lat_similar(Lat1, Lat2, P) :- Lat1 \= Lat2, likelihood(Lat1, Lat2, P).

% MAX aggregation over the 3 prototypes per class (nested lists)
% map over nested lists:
% - Images      : [Img0, Img1, ...]
% - ProtLists   : [[P00,P01,P02], [P10,P11,P12], ...]
% - Dists       : [MaxScore0, MaxScore1, ...]
map_encode_decode_max([], [], []).
map_encode_decode_max([Image|Images], [Prots|Prototypes], [Pmax|Ps]) :-
    % evaluate all three prototypes for this class
    maplist(encode_decode(Image), Prots, Scores),   % closure encode_decode(Image, Prot, P)
    list_max(Scores, Pmax),
    map_encode_decode_max(Images, Prototypes, Ps).

% list maximum
list_max([X], X).
list_max([H|T], M) :- list_max(T, M1), max2(H, M1, M).
max2(A,B,A) :- A >= B.
max2(A,B,B) :- B >  A.

% all_prob/3 using nested prototype lists and max aggregation
all_prob(Images, Classes, Dists) :-
    classes_prototypes(Classes, PrototypesNested),   % [[tensor(C,0),tensor(C,1),tensor(C,2)] ...]
    map_encode_decode_max(Images, PrototypesNested, Dists).
