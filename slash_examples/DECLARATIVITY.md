# Is SLASH declarative?

(NPPs) are joint distributions — is there a declarative formulation of
SLASH? Findings from reading and probing the implementation
(`slash_src/SLASH/slash.py`, `einsum_wrapper.py`):

## The semantics is (almost) declarative at the predicate level

An NPP backed by a probabilistic circuit models the *joint* P(X, C).
The +/− notation annotates, per occurrence of an NPP atom, which
arguments are given, and selects one of four query types
(`replace_plus_minus_occurences`, `slash.py:36-69`):

| notation | query type | computed by the PC |
|---|---|---|
| `digit(0,+X,-C)` | 1 | posterior P(C\|X) |
| `digit(0,-X,+C)` | 2 | likelihood P(X\|C) |
| `digit(0,-X,-C)` | 3 | joint P(X,C) |
| `digit(0,+X,+C)` | 4 | prior P(C) |

So, unlike DeepProbLog-style conditional neural predicates, one trained
NPP serves several query directions — in this sense the *language* is
declarative over the direction of probabilistic conditioning.

## The implementation is not declarative over non-symbolic arguments

The `−` on an image slot never *binds* an image. It only changes which
scalar the PC returns **for a tensor that must still be supplied with
the query**: `SLASH.infer`/`learn` iterate over the input terms
occurring in ground NPP atoms and look each one up in the data
dictionary (`slash.py:400-413`). Verified empirically:

- `infer(dataDic={}, query=':- not seven(i1).')` on the program
  `seven(X) :- digit(0,-X,+7).` fails with `KeyError: 'i1'` — an
  unbound image argument has no grounding path.
- The same query succeeds the moment a concrete tensor is bound to
  `i1`, returning the stable model with `digit(0,2,i1,7)`.

Image *generation* in SLASH (their `mnist_generative` experiment) is an
extra-logical operation: `EiNet.sample(num_samples, class_idx=c)` is
called from Python, outside the program, with the class chosen by the
experimenter. The reasoning engine never decides what to generate.

## What a declarative SLASH would need

The ingredient exists: the PC supports tractable class-conditional
sampling. What is missing is a grounding rule that, on encountering an
NPP atom with an unbound non-symbolic argument, introduces a fresh
tensor term sampled from P(X | C=c) for each enumerated class c — i.e.,
letting resolution, not the experimenter, trigger generation. That is
precisely the mechanism the prototype-based declarative neural
predicates of this repository provide (sample from the prototype, decode,
score); porting it to SLASH would mean teaching `networkAtom2MVPPrules`
to emit sampled-tensor terms for `−` arguments instead of requiring an
input binding.

**Summary: the framework's semantics is declarative over query
direction, but not over argument binding; the implementation requires
every non-symbolic argument to be an input. A declarative formulation is
plausible (the NPP has everything needed) but is a genuine extension,
not a re-formulation.**
