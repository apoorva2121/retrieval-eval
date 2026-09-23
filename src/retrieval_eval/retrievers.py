"""Three retrievers with one interface: rank(query) -> list[(doc_id, score)]."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Retriever:
    def __init__(self, doc_ids: list[str], doc_texts: list[str], k1: float, b: float) -> None:
        self.doc_ids = doc_ids
        self.bm25 = BM25Okapi([tokenize(t) for t in doc_texts], k1=k1, b=b)

    def rank(self, query: str, k: int) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.doc_ids[i], float(scores[i])) for i in top]


class DenseRetriever:
    def __init__(self, doc_ids: list[str], doc_texts: list[str], model: str, batch_size: int) -> None:
        self.doc_ids = doc_ids
        self.model = SentenceTransformer(model)
        self.emb = self.model.encode(doc_texts, batch_size=batch_size, normalize_embeddings=True,
                                     show_progress_bar=True, convert_to_numpy=True).astype("float32")

    def rank(self, query: str, k: int) -> list[tuple[str, float]]:
        q = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")[0]
        scores = self.emb @ q
        top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.doc_ids[i], float(scores[i])) for i in top]


class HybridRRF:
    """Reciprocal rank fusion of BM25 and dense rankings. No score normalisation needed."""

    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever, rrf_k: int, dense_weight: float) -> None:
        self.bm25, self.dense, self.rrf_k, self.w = bm25, dense, rrf_k, dense_weight

    def rank(self, query: str, k: int) -> list[tuple[str, float]]:
        fused: dict[str, float] = {}
        for r, (did, _) in enumerate(self.bm25.rank(query, k)):
            fused[did] = fused.get(did, 0.0) + (1 - self.w) / (self.rrf_k + r + 1)
        for r, (did, _) in enumerate(self.dense.rank(query, k)):
            fused[did] = fused.get(did, 0.0) + self.w / (self.rrf_k + r + 1)
        return sorted(fused.items(), key=lambda kv: -kv[1])[:k]


def build_retrievers(ds, cfg: dict[str, Any]) -> dict[str, Any]:
    r = cfg["retrievers"]
    bm25 = BM25Retriever(ds.doc_ids, ds.doc_texts, **r["bm25"])
    dense = DenseRetriever(ds.doc_ids, ds.doc_texts, r["dense"]["model"], r["dense"]["batch_size"])
    hybrid = HybridRRF(bm25, dense, r["hybrid"]["rrf_k"], r["hybrid"]["dense_weight"])
    return {"BM25": bm25, "Dense": dense, "Hybrid-RRF": hybrid}
