% HWF with prototype-based encodings (two prototype families)
%   - digits:    0..9
%   - operators: +,-,*,/

% Parsign from HWF
almost_equal(X,Y) :- ground(Y), abs(X-Y) < 0.0001.
almost_equal(X,Y) :- var(Y),   Y is float(X).

expression(Images,Result) :-
    detect_all(Images,Symbols),
    parse(Symbols,Result).

detect_all([N],[N2]) :- detect_number(N,N2).
detect_all([N,O|T],[N2,O2|T2]) :-
    detect_number(N,N2),
    detect_operator(O,O2),
    detect_all(T,T2).

parse([N],R) :-
    almost_equal(N,R).

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
    N2 \= 0,
    N3 is N1/N2,
    parse([N3|T],R).

% Prototype families prototype_digit/2  and  prototype_op/2

% Digits 0..9
prototype_digit(D, tensor(prototype_digit(D))) :- between(0,9,D).

% Operators (enumerated)
prototype_op(+, tensor(prototype_op(+))).
prototype_op(-, tensor(prototype_op(-))).
prototype_op(*, tensor(prototype_op(*))).
prototype_op(/, tensor(prototype_op(/))).

% DIGITS: annotated disjunction over 10 classes
P0::detect_number(I0,0) ; P1::detect_number(I1,1) ; P2::detect_number(I2,2) ; P3::detect_number(I3,3) ; P4::detect_number(I4,4) ;  P5::detect_number(I5,5) ; P6::detect_number(I6,6) ; P7::detect_number(I7,7) ; P8::detect_number(I8,8) ; P9::detect_number(I9,9)
 :-
    all_prob_digits(
        [I0,I1,I2,I3,I4,I5,I6,I7,I8,I9],
        [0,1,2,3,4,5,6,7,8,9],
        [P0,P1,P2,P3,P4,P5,P6,P7,P8,P9]
    ).

% OPERATORS: annotated disjunction over + - * /
PA::detect_operator(IA,+) ; PS::detect_operator(IS,-) ; PM::detect_operator(IM,*) ; PD::detect_operator(ID,/)
 :-
    all_prob_ops(
        [IA,IS,IM,ID],
        [+, -, *, /],
        [PA,PS,PM,PD]
    ).

maplist(_, [], []).
maplist(P, [H1|T1], [H2|T2]) :-
    call(P, H1, H2),
    maplist(P, T1, T2).

map_encode_decode([], [], []).
map_encode_decode([Image|Images], [Prot|Prototypes], [P|Probs]) :-
    encode_decode(Image, Prot, P),
    map_encode_decode(Images, Prototypes, Probs).

% Route to the right prototype family
all_prob_digits(Images, Classes, Dists) :-
    maplist(prototype_digit, Classes, Prototypes),
    map_encode_decode(Images, Prototypes, Dists).

all_prob_ops(Images, Classes, Dists) :-
    maplist(prototype_op, Classes, Prototypes),
    map_encode_decode(Images, Prototypes, Dists).

encode_decode(Image, Prototype, P) :-
    encode(Image, Prototype, P1),
    decode(Prototype, Image, P2),
    mul(P1, P2, P).

% Encode: if Image is ground, get latent and compare to Prototype latent;
%         if Image is var, sample from Prototype and decode once to Image, then compare latents.
encode(Image, Prot, P) :-
    ground(Image),
    encoder(Image, Latent),
    lat_similar(Prot, Latent, P).
encode(Image, Prot, P) :-
    var(Image),
    sample(Prot, Sample),
    decoder(Sample, Image),
    lat_similar(Prot, Sample, P).

% Decode: if Prototype is ground, decode its latent to an image and compare images;
%         if Prototype is var, "invert" by encoding the image to its latent.
decode(Prot, Image, P) :-
    ground(Prot),
    sample(Prot, Latent),
    decoder(Latent, Image2),
    im_similar(Image, Image2, P).
decode(Prot, Image, 1.0) :-
    var(Prot),
    encoder(Image, Prot).

% Neural predicates
nn(encoder, [Image], Latent) :: encoder(Image, Latent).
nn(decoder, [Latent], Image) :: decoder(Latent, Image).

% Similarity
im_similar(X, X, 1.0).
im_similar(Image1, Image2, P) :- Image1 \= Image2, mse(Image1, Image2, P).

lat_similar(X, X, 1.0).
lat_similar(Lat1, Lat2, P) :- Lat1 \= Lat2, likelihood(Lat1, Lat2, P).