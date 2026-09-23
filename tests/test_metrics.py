import math

from retrieval_eval.metrics import ndcg_at_k, recall_at_k
from retrieval_eval.retrievers import tokenize


def test_ndcg_perfect_and_worst():
    rels = {"a": 2, "b": 1}
    assert math.isclose(ndcg_at_k(["a", "b", "c"], rels, 10), 1.0)
    assert ndcg_at_k(["x", "y"], rels, 10) == 0.0


def test_ndcg_order_matters():
    rels = {"a": 1}
    assert ndcg_at_k(["a", "b"], rels, 10) > ndcg_at_k(["b", "a"], rels, 10)


def test_recall():
    rels = {"a": 1, "b": 1, "c": 1}
    assert math.isclose(recall_at_k(["a", "z", "b"], rels, 3), 2 / 3)


def test_tokenize_lowercases_and_strips_punct():
    assert tokenize("COVID-19, mRNA!") == ["covid", "19", "mrna"]
