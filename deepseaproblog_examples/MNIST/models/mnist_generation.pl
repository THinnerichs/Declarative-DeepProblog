% DeepSeaProbLog MNIST baseline program, adapted from the official
% LOGICVAE example (operation_generation.pl) of
% https://github.com/ML-KULeuven/deepseaproblog (De Smet et al., UAI 2023).
% Subtraction is replaced by addition to match the add/3 task, and a
% single-digit variant is added for the digit/2 task.

% Variables
shape(X, S) ~ normal(encoder_net([X])).
generation(Latent, Condition, Gen) ~ vae_decoder(decoder_net([Latent, Condition])).
digit(X, Y) ~ categorical(mnist_class([X]), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).
prior(ID, S) ~ normal([[0, 0, 0, 0], [1, 1, 1, 1]]).

% Program
%% Direct supervision (digit/2) with auto-encoding
encode_decode_digit(Image, N) :-
    digit(Image, D), tf_eq(N, D),
    argmax(D, P),
    shape(Image, Shape),
    generation(Shape, P, Gen),
    prior(_, Prior),
    equals(Shape, Prior),
    unification(Gen, Image).

%% Curriculum (labelled pairs, as in the original example)
image_addition_curriculum(Image1, Image2, N1, N2) :-
    digit(Image1, D1), digit(Image2, D2),
    tf_eq(N1, D1), tf_eq(N2, D2).

%% Distant supervision (add/3) with auto-encoding
encode_decode_addition(Image1, Image2, Sum) :-
    digit(Image1, D1), digit(Image2, D2),
    tf_add(D1, D2, R), tf_eq(Sum, R),
    argmax(D1, P1), argmax(D2, P2),
    shape(Image1, Shape1), shape(Image2, Shape2),
    generation(Shape1, P1, Gen1), generation(Shape2, P2, Gen2),
    prior(_, Prior1), prior(_, Prior2),
    equals(Shape1, Prior1), equals(Shape2, Prior2),
    unification(Gen1, Image1), unification(Gen2, Image2).

%% Classification-only queries for evaluation
image_digit(Image, N) :-
    digit(Image, D), tf_eq(N, D).

image_addition(Image1, Image2, Sum) :-
    digit(Image1, D1), digit(Image2, D2),
    tf_add(D1, D2, R), tf_eq(Sum, R).

%% Generation logic
generate_digit(N, Gen) :-
    member(N, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    prior(_, Prior),
    generation(Prior, N, Gen).

generate_addition(Sum, GenLeft, GenRight) :-
    member(D1, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), member(D2, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    prior(_, Prior1), prior(_, Prior2),
    Sum is D1 + D2,
    generation(Prior1, D1, GenLeft), generation(Prior2, D2, GenRight).

%% multi_add/9 generation analogue: two 4-digit numbers from their sum.
%% Enumerates all digit combinations symbolically; expected to be
%% intractable (10^8 groundings).
generate_multi_addition(Sum, G1, G2, G3, G4, G5, G6, G7, G8) :-
    member(D1, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), member(D2, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    member(D3, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), member(D4, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    member(D5, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), member(D6, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    member(D7, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), member(D8, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    Sum is 1000*D1 + 100*D2 + 10*D3 + D4 + 1000*D5 + 100*D6 + 10*D7 + D8,
    prior(_, P1), prior(_, P2), prior(_, P3), prior(_, P4),
    prior(_, P5), prior(_, P6), prior(_, P7), prior(_, P8),
    generation(P1, D1, G1), generation(P2, D2, G2),
    generation(P3, D3, G3), generation(P4, D4, G4),
    generation(P5, D5, G5), generation(P6, D6, G6),
    generation(P7, D7, G7), generation(P8, D8, G8).
