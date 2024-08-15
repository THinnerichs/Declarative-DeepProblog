% nn(net1,[X],Y,[0,1,2,3,4,5,6,7,8,9]) :: detect_number(X,Y).
% nn(net2,[X],Y,[+,-,*,/]) :: detect_operator(X,Y).

detect_all([N],[N2]) :- detect_number(N,N2).
detect_all([N,O|T],[N2,O2|T2]) :- detect_number(N,N2), detect_operator(O,O2), detect_all(T,T2).

almost_equal(X,Y) :- ground(Y),abs(X-Y) < 0.0001.
almost_equal(X,Y) :- var(Y), Y is float(X).

expression(Images,Result) :- detect_all(Images,Symbols),parse(Symbols,Result).

parse([N],R) :-almost_equal(N,R).

parse([N1,+|T], R) :-
    parse(T,R2),
    almost_equal(N1+R2,R).

parse([N1,-|T], R) :-
    parse([-1,*|T],R2),
    almost_equal(N1+R2,R).

parse([N1,*,N2|T], R) :-
    N3 is N1*N2,
    parse([N3|T],R).

parse([N1,/,N2|T], R) :-
    N2 \== 0,
    N3 is N1/N2,
    parse([N3|T],R).


% Declarative logic
num_prototype(X, tensor(num_prototype(X))) :- between(0,9,X).
op_prototype(X, tensor(op_prototype([X]))) :- member(X, [+,-,*,/]).


% detect_number(Image, Digit) :- num_prototype(Digit, Prototype), encode_decode(Image, Prototype).
% detect_operator(Image, Op) :- op_prototype(Op, Prototype), encode_decode(Image, Prototype).

P0::detect_number(I,0) ; P1::detect_number(I,1); P2::detect_number(I,2); P3::detect_number(I,3); P4::detect_number(I,4); P5::detect_number(I,5); P6::detect_number(I,6); P7::detect_number(I,8); P9::detect_number(I,9):- all_prob_num(I,[0,1,2,3,4,5,6,7,8,9],[P0, P1, P2, P3, P4, P5, P6, P7, P8, P9]).

P0::detect_operator(I,+) ; P1::detect_operator(I,-); P2::detect_operator(I,*); P3::detect_operator(I,/):- all_prob_op(I,[+,-,*,/],[P0, P1, P2, P3]).

maplist(_, [], []).
maplist(P, [H1|T1], [H2|T2]) :-
    call(P, H1, H2),
    maplist(P, T1, T2).

all_prob_num(Image,Classes, NormDists) :- maplist(num_prototype,Classes,Prototypes), map_encode_decode(Image, Prototypes, NormDists).%, norm(Dists, NormDists), writeln('BUmm2').
all_prob_op(Image,Classes, NormDists) :- maplist(op_prototype,Classes,Prototypes), map_encode_decode(Image, Prototypes, NormDists).%, norm(Dists, NormDists), writeln('BUmm2').


map_encode_decode(Image, [], []).
map_encode_decode(Image, [Prot|Prototypes], [P|Probs]) :- encode_decode(Image, Prot, P), map_encode_decode(Image, Prototypes, Probs).

encode_decode(Image, Prototype, P) :- encode(Image, Prototype, P1), decode(Prototype, Image, P2), mul(P1, P2, P). 

encode(Image, Latent, P) :- ground(Image), encoder(Image,Latent2), lat_similar(Latent, Latent2, P).
encode(Image, Latent, 1.0) :- var(Image), decoder(Latent, Image).

decode(Latent, Image, P) :- ground(Latent), decoder(Latent, Image2), im_similar(Image, Image2, P).
decode(Latent, Image, 1.0) :- var(Latent), encoder(Image, Latent).

nn(encoder, [Image], Latent) :: encoder(Image, Latent).
nn(decoder, [Latent], Image) :: decoder(Latent, Image).

im_similar(X,X, 1.0).
im_similar(Image1, Image2, P) :- Image1 \= Image2, mse(Image1, Image2, P).

lat_similar(X,X, 1.0).
lat_similar(Lat1, Lat2, P) :- Lat1 \= Lat2, distrcos(Lat1, Lat2, P).
% lat_similar(X,X).
% lat_similar(Lat1, Lat2, P) :- Lat1 \= Lat2, likelihood(Lat1, Lat2, P).
