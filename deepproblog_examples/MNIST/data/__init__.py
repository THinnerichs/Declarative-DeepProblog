import itertools
import json
import random
from pathlib import Path
from typing import Callable, List, Iterable, Tuple

import torchvision
import torchvision.transforms as transforms
from problog.logic import Term, list2term, Constant
from torch.utils.data import Dataset as TorchDataset
from torch import tensor

from deepproblog.dataset import Dataset
from deepproblog.query import Query

_DATA_ROOT = Path(__file__).parent

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

datasets = {
    "train": torchvision.datasets.MNIST(
        root=str(_DATA_ROOT), train=True, download=True, transform=transform
    ),
    "test": torchvision.datasets.MNIST(
        root=str(_DATA_ROOT), train=False, download=True, transform=transform
    ),
}


def digits_to_number(digits: Iterable[int]) -> int:
    number = 0
    for d in digits:
        number *= 10
        number += d
    return number

def list2term(xs):
    """Helper: Python list -> Prolog list term."""
    t = Term("[]")
    for x in reversed(xs):
        t = Term(".", x, t)
    return t


class MNIST_Images(object):
    def __init__(self, subset):
        self.subset = subset

    def __getitem__(self, item):
        return datasets[self.subset][int(item[0])][0]


MNIST_train = MNIST_Images("train")
MNIST_test = MNIST_Images("test")


class MNIST(Dataset):
    def __len__(self):
        return len(self.data)

    def to_query(self, i):
        l = Constant(self.data[i][1])
        return Query(
            Term("digit", Term("tensor", Term(self.dataset, Term("a"))), l),
            substitution={Term("a"): Constant(i)},
        )

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.data = datasets[dataset]

    def get_labels(self):
        labels = []
        for i in range(len(self.data)):
            labels.append(self.data[i][1])
        return tensor(labels)

def addition(n: int, dataset: str, seed=None):
    """Returns a dataset for binary addition"""
    return MNISTOperator(
        dataset_name=dataset,
        function_name="addition" if n == 1 else "multi_addition",
        operator=sum,
        size=n,
        arity=2,
        seed=seed,
    )


class MNISTOperator(Dataset, TorchDataset):
    def __getitem__(self, index: int) -> Tuple[list, list, int]:
        l1, l2 = self.data[index]
        label = self._get_label(index)
        l1 = [self.dataset[x][0] for x in l1]
        l2 = [self.dataset[x][0] for x in l2]
        return l1, l2, label

    def __init__(
        self,
        dataset_name: str,
        function_name: str,
        operator: Callable[[List[int]], int],
        size=1,
        arity=2,
        seed=None,
    ):
        """Generic dataset for operator(img, img) style datasets.

        :param dataset_name: Dataset to use (train, val, test)
        :param function_name: Name of Problog function to query.
        :param operator: Operator to generate correct examples
        :param size: Size of numbers (number of digits)
        :param arity: Number of arguments for the operator
        :param seed: Seed for RNG
        """
        super(MNISTOperator, self).__init__()
        assert size >= 1
        assert arity >= 1
        self.dataset_name = dataset_name
        self.dataset = datasets[self.dataset_name]
        self.function_name = function_name
        self.operator = operator
        self.size = size
        self.arity = arity
        self.seed = seed
        mnist_indices = list(range(len(self.dataset)))
        if seed is not None:
            rng = random.Random(seed)
            rng.shuffle(mnist_indices)
        dataset_iter = iter(mnist_indices)
        # Build list of examples (mnist indices)
        self.data = []
        try:
            while dataset_iter:
                self.data.append(
                    [
                        [next(dataset_iter) for _ in range(self.size)]
                        for _ in range(self.arity)
                    ]
                )
        except StopIteration:
            pass

    def to_file_repr(self, i):
        """Old file represenation dump. Not a very clear format as multi-digit arguments are not separated"""
        return f"{tuple(itertools.chain(*self.data[i]))}\t{self._get_label(i)}"

    def to_json(self):
        """
        Convert to JSON, for easy comparisons with other systems.

        Format is [EXAMPLE, ...]
        EXAMPLE :- [ARGS, expected_result]
        ARGS :- [MULTI_DIGIT_NUMBER, ...]
        MULTI_DIGIT_NUMBER :- [mnist_img_id, ...]
        """
        data = [(self.data[i], self._get_label(i)) for i in range(len(self))]
        return json.dumps(data)

    def to_query(self, i: int) -> Query:
        """Generate queries"""
        mnist_indices = self.data[i]
        expected_result = self._get_label(i)

        # Build substitution dictionary for the arguments
        subs = dict()
        var_names = []
        for i in range(self.arity):
            inner_vars = []
            for j in range(self.size):
                t = Term(f"p{i}_{j}")
                subs[t] = Term(
                    "tensor",
                    Term(
                        self.dataset_name,
                        Constant(mnist_indices[i][j]),
                    ),
                )
                inner_vars.append(t)
            var_names.append(inner_vars)

        # print("var_names:", var_names)
        # print("subs", subs)

        # Build query
        if self.size == 1:
            return Query(
                Term(
                    self.function_name,
                    *(e[0] for e in var_names),
                    Constant(expected_result),
                ),
                subs,
            )
        else:
            return Query(
                Term(
                    self.function_name,
                    *(list2term(e) for e in var_names),
                    Constant(expected_result),
                ),
                subs,
            )

    def _get_label(self, i: int):
        mnist_indices = self.data[i]
        # Figure out what the ground truth is, first map each parameter to the value:
        ground_truth = [
            digits_to_number(self.dataset[j][1] for j in i) for i in mnist_indices
        ]
        # Then compute the expected value:
        expected_result = self.operator(ground_truth)
        return expected_result

    def get_labels(self):
        return tensor([self._get_label(i) for i in range(len(self))])

    def __len__(self):
        return len(self.data)

# ----------- Scallop datasets ---------------------

class MNISTNot34Binary(Dataset):
    """
    Each example is a SINGLE image; label is 1 if digit ∉ {3,4}, else 0.
    """
    def __init__(self, dataset_name: str, seed: int | None = None):
        super().__init__()
        assert dataset_name in datasets
        self.dataset_name = dataset_name
        self.dataset = datasets[dataset_name]
        self.indices = list(range(len(self.dataset)))
        if seed is not None:
            rng = random.Random(seed); rng.shuffle(self.indices)

    def __len__(self): return len(self.indices)

    def _get_label(self, i: int) -> int:
        idx = self.indices[i]
        _, digit = self.dataset[idx]
        return 1 if digit not in (3,4) else 0

    def get_labels(self):
        return tensor([self._get_label(i) for i in range(len(self))])

    def to_query(self, i: int) -> Query:
        idx = self.indices[i]
        label = self._get_label(i)
        # Build substitution: ImageVar -> tensor(dataset_name, idx)
        img_var = Term("img")
        subs = {img_var: Term("tensor", Term(self.dataset_name, Constant(idx)))}
        # Query: not_3_or_4(img, Label)
        return Query(Term("not_3_or_4", img_var, Constant(label)), subs)

class MNISTLessThanBinary(Dataset):
    """
    Each example is a PAIR of single-digit images (x, y).
    Label is 1 if digit(x) < digit(y), else 0.
    Prolog: less_than(ImageX, ImageY, Label)
    """
    def __init__(self, dataset_name: str, seed: int | None = None):
        super().__init__()
        self.dataset_name = dataset_name
        self.dataset = datasets[dataset_name]
        idxs = list(range(len(self.dataset)))
        rng = random.Random(seed) if seed is not None else random
        rng.shuffle(idxs)
        # build disjoint pairs (p0,p1), (p2,p3), ...
        self.pairs: list[tuple[int,int]] = [
            (idxs[i], idxs[i+1]) for i in range(0, len(idxs)-1, 2)
        ]

    def __len__(self): return len(self.pairs)

    def _get_label(self, i: int) -> int:
        a, b = self.pairs[i]
        da = int(self.dataset[a][1]); db = int(self.dataset[b][1])
        return 1 if da < db else 0

    def get_labels(self):
        return tensor([self._get_label(i) for i in range(len(self))])

    def to_query(self, i: int) -> Query:
        a, b = self.pairs[i]
        y = self._get_label(i)
        XA, XB = Term("xa"), Term("xb")
        subs = {
            XA: Term("tensor", Term(self.dataset_name, Constant(a))),
            XB: Term("tensor", Term(self.dataset_name, Constant(b))),
        }
        # less_than(xa, xb, Label)
        return Query(Term("less_than", XA, XB, Constant(y)), subs)

class MNISTCount3s(Dataset):
    """
    Each example is a LIST of k single-digit images; label is count of '3' digits.
    """
    def __init__(self, dataset_name: str, list_len: int = 5, seed: int | None = None):
        super().__init__()
        assert list_len >= 1
        self.dataset_name = dataset_name
        self.dataset = datasets[dataset_name]
        self.list_len = list_len
        idxs = list(range(len(self.dataset)))
        if seed is not None:
            rng = random.Random(seed); rng.shuffle(idxs)
        else:
            random.shuffle(idxs)
        self.data: List[List[int]] = [idxs[i:i+list_len] for i in range(0, len(idxs)-list_len+1, list_len)]

    def __len__(self): return len(self.data)

    def _get_label(self, i: int) -> int:
        idxs = self.data[i]
        digits = [int(self.dataset[j][1]) for j in idxs]
        return sum(1 for d in digits if d == 3)

    def get_labels(self):
        return tensor([self._get_label(i) for i in range(len(self))])

    def to_query(self, i: int) -> Query:
        idxs = self.data[i]
        label = self._get_label(i)
        vars_ = []
        subs = {}
        for k, idx in enumerate(idxs):
            v = Term(f"p{k}")
            vars_.append(v)
            subs[v] = Term("tensor", Term(self.dataset_name, Constant(idx)))
        return Query(Term("count_digit_3", list2term(vars_), Constant(label)), subs)


class MNISTCount34(Dataset):
    """
    Each example is a LIST of k single-digit images.
    Label is the count of digits that are in {3,4}.
    Prolog: count_3_or_4([Imgs], Count)
    """
    def __init__(self, dataset_name: str, list_len: int = 5, seed: int | None = None):
        super().__init__()
        assert list_len >= 1
        self.dataset_name = dataset_name
        self.dataset = datasets[dataset_name]
        idxs = list(range(len(self.dataset)))
        rng = random.Random(seed) if seed is not None else random
        rng.shuffle(idxs)
        self.data: list[list[int]] = [
            idxs[i:i+list_len] for i in range(0, len(idxs)-list_len+1, list_len)
        ]

    def __len__(self): return len(self.data)

    def _get_label(self, i: int) -> int:
        idxs = self.data[i]
        digits = [int(self.dataset[j][1]) for j in idxs]
        return sum(1 for d in digits if d in (3, 4))

    def get_labels(self):
        return tensor([self._get_label(i) for i in range(len(self))])

    def to_query(self, i: int) -> Query:
        idxs = self.data[i]
        y = self._get_label(i)
        vars_, subs = [], {}
        for k, idx in enumerate(idxs):
            v = Term(f"p{k}")
            vars_.append(v)
            subs[v] = Term("tensor", Term(self.dataset_name, Constant(idx)))
        return Query(Term("count_3_or_4", list2term(vars_), Constant(y)), subs)



class _MNISTSumK(Dataset):
    """
    Each example is a LIST of exactly K single-digit images.
    Label is the arithmetic sum of the digits.
    Prolog: sumK([Imgs], Sum) where K in {2,3,4}
    """
    PREDICATE = None  # override in subclass: "sum2" | "sum3" | "sum4"

    def __init__(self, dataset_name: str, K: int, seed: int | None = None):
        super().__init__()
        assert dataset_name in datasets
        assert K in (2,3,4)
        self.dataset_name = dataset_name
        self.dataset = datasets[dataset_name]
        self.K = K

        idxs = list(range(len(self.dataset)))
        rng = random.Random(seed) if seed is not None else random
        rng.shuffle(idxs)
        # slice into chunks of length K
        self.data: list[list[int]] = [idxs[i:i+K] for i in range(0, len(idxs) - K + 1, K)]

    def __len__(self): return len(self.data)

    def _get_label(self, i: int) -> int:
        idxs = self.data[i]
        digits = [int(self.dataset[j][1]) for j in idxs]
        return sum(digits)

    def get_labels(self):
        return tensor([self._get_label(i) for i in range(len(self))])

    def to_query(self, i: int) -> Query:
        idxs = self.data[i]
        y = self._get_label(i)
        vars_, subs = [], {}
        for k, idx in enumerate(idxs):
            v = Term(f"p{k}")
            vars_.append(v)
            subs[v] = Term("tensor", Term(self.dataset_name, Constant(idx)))
        # sumK([p0,...,p{K-1}], y)
        return Query(Term(self.PREDICATE, list2term(vars_), Constant(y)), subs)

class MNISTSum2(_MNISTSumK):
    PREDICATE = "sum2"
    def __init__(self, dataset_name: str, seed: int | None = None):
        super().__init__(dataset_name, K=2, seed=seed)

class MNISTSum3(_MNISTSumK):
    PREDICATE = "sum3"
    def __init__(self, dataset_name: str, seed: int | None = None):
        super().__init__(dataset_name, K=3, seed=seed)

class MNISTSum4(_MNISTSumK):
    PREDICATE = "sum4"
    def __init__(self, dataset_name: str, seed: int | None = None):
        super().__init__(dataset_name, K=4, seed=seed)


