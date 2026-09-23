"""Evaluate BM25 / dense / hybrid on each configured BEIR dataset; write per-query results.

Usage:
    python -m retrieval_eval.run [--config config.yaml] [--datasets scifact nfcorpus]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .data import load_beir, load_config
from .metrics import ndcg_at_k, recall_at_k
from .retrievers import build_retrievers


def evaluate_dataset(ds, retrievers, cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    e = cfg["eval"]
    rows = []
    for qid, qtext in tqdm(ds.queries.items(), desc=ds.name, leave=False):
        rels = ds.qrels[qid]
        for name, r in retrievers.items():
            ranked = [d for d, _ in r.rank(qtext, e["candidate_depth"])]
            rows.append({"dataset": ds.name, "qid": qid, "retriever": name,
                         "ndcg@10": ndcg_at_k(ranked, rels, e["k_ndcg"]),
                         "recall@100": recall_at_k(ranked, rels, e["k_recall"]),
                         "n_relevant": len(rels), "query": qtext})
    per_query = pd.DataFrame(rows)
    summary = per_query.groupby("retriever")[["ndcg@10", "recall@100"]].mean().sort_values("ndcg@10", ascending=False)
    summary["n_queries"] = per_query.groupby("retriever").size()
    return per_query, summary


def per_query_analysis(pq: pd.DataFrame, n: int) -> dict:
    """Where does hybrid beat BM25, where does it lose, and what do the hardest queries look like."""
    wide = pq.pivot_table(index="qid", columns="retriever", values="ndcg@10")
    wide["query"] = pq.drop_duplicates("qid").set_index("qid")["query"]
    diff = (wide["Hybrid-RRF"] - wide["BM25"])
    out = {
        "hybrid_wins_vs_bm25": int((diff > 0).sum()), "hybrid_ties": int((diff == 0).sum()),
        "hybrid_losses_vs_bm25": int((diff < 0).sum()),
        "dense_wins_vs_bm25": int((wide["Dense"] > wide["BM25"]).sum()),
        "most_improved_by_hybrid": wide.assign(delta=diff).nlargest(n, "delta")[["query", "BM25", "Dense", "Hybrid-RRF"]].round(3).to_dict(orient="records"),
        "hurt_most_by_hybrid": wide.assign(delta=diff).nsmallest(n, "delta")[["query", "BM25", "Dense", "Hybrid-RRF"]].round(3).to_dict(orient="records"),
        "unanswered_by_all": wide[(wide[["BM25", "Dense", "Hybrid-RRF"]].max(axis=1) == 0)][["query"]].head(n).to_dict(orient="records"),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--datasets", nargs="*", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    rep = Path(cfg["paths"]["reports"]); rep.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    wanted = {d["name"]: d for d in cfg["datasets"]}
    names = args.datasets or list(wanted)
    all_summary, all_pq, analysis = [], [], {}
    for name in names:
        t = time.time()
        ds = load_beir(name, cfg["paths"]["data"])
        print(f"\n[{name}] corpus {len(ds.doc_ids):,} docs | {len(ds.queries)} test queries")
        retrievers = build_retrievers(ds, cfg)
        pq, summary = evaluate_dataset(ds, retrievers, cfg)
        summary.insert(0, "dataset", name)
        summary["published_bm25_ndcg10"] = wanted[name]["published_bm25_ndcg10"]
        print(summary.round(4).to_string())
        analysis[name] = per_query_analysis(pq, cfg["eval"]["n_examples"])
        print(f"  hybrid vs BM25 per query: {analysis[name]['hybrid_wins_vs_bm25']} wins / "
              f"{analysis[name]['hybrid_ties']} ties / {analysis[name]['hybrid_losses_vs_bm25']} losses  "
              f"({time.time() - t:.0f}s)")
        all_summary.append(summary); all_pq.append(pq)

    summary = pd.concat(all_summary)
    pq = pd.concat(all_pq)
    summary.to_csv(rep / "summary.csv")
    pq.to_csv(rep / "per_query.csv", index=False)
    (rep / "analysis.json").write_text(json.dumps(analysis, indent=2))
    (rep / "results.json").write_text(json.dumps({
        "summary": summary.reset_index().to_dict(orient="records"),
        "runtime_seconds": round(time.time() - t0, 1), "config": cfg}, indent=2, default=str))
    print(f"\nreports -> {rep}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
