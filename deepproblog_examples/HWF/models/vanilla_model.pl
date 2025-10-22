% nn(net1,[X],Y,[0,1,2,3,4,5,6,7,8,9]) :: detect_number(X,Y).
% nn(net2,[X],Y,[+,-,*,/]) :: detect_operator(X,Y).
% 
% detect_all([N],[N2]) :- detect_number(N,N2).
% detect_all([N,O|T],[N2,O2|T2]) :- detect_number(N,N2), detect_operator(O,O2), detect_all(T,T2).
% 
% almost_equal(X,Y) :- ground(Y),abs(X-Y) < 0.0001.
% almost_equal(X,Y) :- var(Y), Y is float(X).
% 
% expression(Images,Result) :- detect_all(Images,Symbols),parse(Symbols,Result).
% 
% parse([N],R) :-almost_equal(N,R).
% 
% parse([N1,+|T], R) :-
%     parse(T,R2),
%     almost_equal(N1+R2,R).
% 
% parse([N1,-|T], R) :-
%     parse([-1,*|T],R2),
%     almost_equal(N1+R2,R).
% 
% parse([N1,*,N2|T], R) :-
%     N3 is N1*N2,
%     parse([N3|T],R).
% 
% parse([N1,/,N2|T], R) :-
%     N2 \== 0,
%     N3 is N1/N2,
%     parse([N3|T],R).



%%% DeepStochlog inspired parsing:
% If your operator labels are atoms: plus, minus, times, div
nn(net1,[X],Y,[0,1,2,3,4,5,6,7,8,9]) :: detect_number(X,Y).
nn(net2,[X],Op,[+,-,*,/]) :: detect_operator(X,Op).


almost_equal(X,Y) :- ground(Y), abs(X-Y) < 0.0001.
almost_equal(X,Y) :- var(Y),   Y is float(X).

is_number(N, [Img|Rest], Rest) :-
    detect_number(Img, N).

operator(Op, [Img|Rest], Rest) :-
    detect_operator(Img, Op).

factor(N, In, Out) :-
    is_number(N, In, Out).

term(T, In, Out) :-
    factor(F, In, Mid),
    term_r(F, T, Mid, Out).

term_r(Acc, T, In, Out) :-
    operator(*, In, Mid1),
    factor(F, Mid1, Mid2),
    Acc1 is Acc * F,
    term_r(Acc1, T, Mid2, Out).
term_r(Acc, T, In, Out) :-
    operator(/, In, Mid1),
    factor(F, Mid1, Mid2),
    F =\= 0,
    Acc1 is Acc / F,
    term_r(Acc1, T, Mid2, Out).
term_r(Acc, Acc, In, In).  % epsilon

expression(E, In, Out) :-
    term(T, In, Mid),
    expression_r(T, E, Mid, Out).

expression_r(Acc, E, In, Out) :-
    operator(+, In, Mid1),
    term(T, Mid1, Mid2),
    Acc1 is Acc + T,
    expression_r(Acc1, E, Mid2, Out).
expression_r(Acc, E, In, Out) :-
    operator(-, In, Mid1),
    term(T, Mid1, Mid2),
    Acc1 is Acc - T,
    expression_r(Acc1, E, Mid2, Out).
expression_r(Acc, Acc, In, In).  % epsilon

% Direct numeric result
evaluate_images(Images, Value) :-
    expression(Value, Images, []).

% With tolerant unification like your original 'expression/2'
expression(Images, Result) :-
    evaluate_images(Images, Value),
    almost_equal(Value, Result).
