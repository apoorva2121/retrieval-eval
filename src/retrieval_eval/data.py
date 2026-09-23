"""Load a BEIR dataset (corpus, test queries, qrels) from the parquet files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@dataclass
class BeirDataset:
    name: str
    doc_ids: list[str]
    doc_texts: list[str]
    queries: dict[str, str]                    # qid -> text (test split only)
    qrels: dict[str, dict[str, int]]           # qid -> {doc_id: relevance}


def load_beir(name: str, data_dir: str | Path) -> BeirDataset:
    d = Path(data_dir)
    corpus = pd.read_parquet(d / f"{name}_corpus.parquet")
    queries = pd.read_parquet(d / f"{name}_queries.parquet")
    qrels = pd.read_parquet(d / f"{name}_qrels_test.parquet")

    # BEIR convention: title + text
    texts = (corpus["title"].fillna("") + " " + corpus["text"].fillna("")).str.strip().tolist()
    doc_ids = corpus["_id"].astype(str).tolist()

    rel: dict[str, dict[str, int]] = {}
    for qid, did, score in zip(qrels["query-id"].astype(str), qrels["corpus-id"].astype(str), qrels["score"]):
        if int(score) > 0:
            rel.setdefault(qid, {})[did] = int(score)

    qmap = dict(zip(queries["_id"].astype(str), queries["text"]))
    test_queries = {qid: qmap[qid] for qid in rel if qid in qmap}
    return BeirDataset(name=name, doc_ids=doc_ids, doc_texts=texts, queries=test_queries, qrels=rel)
