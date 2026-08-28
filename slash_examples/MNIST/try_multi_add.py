"""Attempt the multi_add/9 generative query in SLASH, under a wall-clock
time limit.

To answer "generate eight digit images such that the two 4-digit numbers
sum to S", SLASH's ASP side must enumerate the stable models of the
ground program (8 neural-probabilistic atoms with 10 values each = 10^8
candidate worlds) that are consistent with the observation, before any
class-conditional sampling can happen. This script runs exactly that
enumeration step (MVPP.find_all_SM_under_query, the same call SLASH's
learning/inference uses) and reports whether it completes.

Run:  python try_multi_add.py    (prints COMPLETED or is killed by the
      wrapper timeout; see run output / results/multi_add_attempt.log)
"""
import sys
import time
from pathlib import Path

_SRC = Path(__file__).parent.parent / "slash_src"
sys.path.insert(0, str(_SRC / "SLASH"))

from mvpp import MVPP

TARGET_SUM = 2764

# ground npp atoms use SLASH's internal form digit(E, T, input, value),
# with T the query-type index introduced by the +/- notation
digit_rules = "\n".join(
    "; ".join(f"@0.1 digit(0,1,i{k},{v})" for v in range(10)) + "."
    for k in range(1, 9)
)

program = digit_rules + """
multi_add(S) :- digit(0,1,i1,D1), digit(0,1,i2,D2), digit(0,1,i3,D3), digit(0,1,i4,D4),
                digit(0,1,i5,D5), digit(0,1,i6,D6), digit(0,1,i7,D7), digit(0,1,i8,D8),
                S=1000*D1+100*D2+10*D3+D4+1000*D5+100*D6+10*D7+D8.
"""

if __name__ == "__main__":
    mvpp = MVPP(program)
    print(f"Enumerating stable models for multi_add({TARGET_SUM})...", flush=True)
    t0 = time.time()
    models = mvpp.find_all_SM_under_query(f":- not multi_add({TARGET_SUM}).")
    print(f"COMPLETED in {time.time() - t0:.1f}s with {len(models)} stable models")
