"""Attempt the multi_add/9 generative query with a trained
DeepSeaProbLog addition model, under a wall-clock time limit.

The query enumerates all 10^8 digit-pair combinations symbolically, so
grounding/compilation is expected to be intractable; this script
documents that outcome honestly instead of asserting it.
"""
import signal
import sys
import time
from pathlib import Path

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent / "deepseaproblog"))
sys.path.insert(0, str(_HERE))

from problog.logic import Constant, Var

from run_mnist_dsp import build_model

TIME_LIMIT_S = 1800


class Timeout(Exception):
    pass


def _raise_timeout(signum, frame):
    raise Timeout()


def main():
    model, classifier, encoder, decoder = build_model(0)
    weight_dir = _HERE / "saved_networks" / "addition_seed0"
    # networks need a build pass before loading weights
    import tensorflow as tf

    classifier.call(tf.zeros([1, 28, 28, 1]))
    enc_out = encoder.call(tf.zeros([1, 28, 28, 1]))
    decoder.call(tf.zeros([1, 4]), tf.constant([[0.0] * 10]))
    classifier.load_weights(str(weight_dir / "classifier"))
    encoder.load_weights(str(weight_dir / "encoder"))
    decoder.load_weights(str(weight_dir / "decoder"))

    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.alarm(TIME_LIMIT_S)
    t0 = time.time()
    try:
        result = model.solve_query(
            "generate_multi_addition",
            [Constant(2764)] + [Var(f"X{i}") for i in range(8)],
            generate=True,
        )
        elapsed = time.time() - t0
        n = len(list(result[0].result.keys()))
        print(f"COMPLETED in {elapsed:.1f}s with {n} groundings")
    except Timeout:
        print(f"TIMEOUT after {TIME_LIMIT_S}s (grounding/compilation did not finish)")
    except MemoryError:
        print(f"OUT OF MEMORY after {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
