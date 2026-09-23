"""nDCG@k and Recall@k from the TREC / BEIR definitions (graded relevance, log2 discount)."""

from __future__ import annotations

import math


def ndcg_at_k(ranked: list[str], rels: dict[str, int], k: int) -> float:
    dcg = 0.0
    for i, did in enumerate(ranked[:k]):
        g = rels.get(did, 0)
        if g > 0:
            dcg += (2 ** g - 1) / math.log2(i + 2)
    ideal = sorted(rels.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked: list[str], rels: dict[str, int], k: int) -> float:
    if not rels:
        return 0.0
    hit = sum(1 for did in ranked[:k] if did in rels)
    return hit / len(rels)
