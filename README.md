# retrieval-eval

Honest retrieval evaluation for RAG: **BM25 vs dense embeddings vs hybrid
(reciprocal rank fusion)** on two BEIR benchmarks, scored with nDCG@10 and
Recall@100 implemented from the TREC definitions and checked against the
published BM25 numbers — plus a **per-query analysis** of where each method
wins and, more usefully, where it fails.

| Dataset | Retriever | nDCG@10 | Recall@100 | Published BM25 (BEIR) |
|---|---|---|---|---|
| **SciFact** (5,183 docs, 300 queries) | **Hybrid-RRF** | **0.687** | **0.957** | 0.665 |
| | BM25 (rank_bm25, k1 0.9, b 0.4) | 0.649 | 0.876 | |
| | Dense (all-MiniLM-L6-v2, 22M) | 0.645 | 0.925 | |
| **NFCorpus** (3,633 docs, 323 queries) | **Hybrid-RRF** | **0.336** | **0.318** | 0.325 |
| | Dense | 0.317 | 0.312 | |
| | BM25 | 0.306 | 0.244 | |

> **Where the numbers come from.** Official BEIR test qrels, full corpora, my own
> metric code. My BM25 lands 0.016–0.019 below the published figure on both
> datasets — the published runs use Anserini/Lucene BM25 with its tokenizer and
> stemming; mine is `rank_bm25` with a plain alphanumeric tokenizer. The gap is
> consistent in sign and size across both datasets, which is what you expect
> from a tokenizer difference and not from a metric bug. Hybrid beats the
> *published* BM25 on both datasets regardless.

## Why this exists

Every RAG demo shows an answer. Almost none show whether the retriever found
the right passage — and when the retriever misses, the generator confidently
answers from the wrong context. This repo is the retrieval half of an
evaluation harness: it measures the retriever alone, on public benchmarks with
graded relevance judgements, and asks the questions a RAG owner needs answered
before touching a prompt:

1. Does dense retrieval actually beat lexical on my kind of query?
2. Does fusing them help, and on *which* queries does it hurt?
3. What does a query that no retriever can answer look like?

## Workflow

```mermaid
flowchart LR
    A[BEIR corpus + queries + test qrels<br/>SciFact · NFCorpus] --> B1[BM25<br/>rank_bm25 · k1 0.9 · b 0.4]
    A --> B2[Dense<br/>all-MiniLM-L6-v2<br/>normalised · dot product]
    B1 --> C[Hybrid RRF<br/>1 / 60 + rank<br/>equal weights]
    B2 --> C
    B1 --> D[Top-100 per query]
    B2 --> D
    C --> D
    D --> E[nDCG@10 · Recall@100<br/>TREC definitions, graded]
    E --> F[summary.csv]
    E --> G[per_query.csv<br/>analysis.json<br/>wins · losses · unanswerable]
```

## What I did

### 1. Data — `data.py`
BEIR parquet exports of corpus (`title + text`), queries, and **test-split qrels**
with graded relevance. Only queries that have at least one relevant document
are evaluated (BEIR convention).

### 2. Metrics — `metrics.py`
`nDCG@10` with `(2^rel − 1) / log2(rank + 1)` gain and ideal-DCG normalisation;
`Recall@100` as fraction of relevant documents retrieved. Both written from the
definitions, tested on hand-constructed cases, and sanity-checked against the
published BM25 numbers (see the note above).

### 3. Retrievers — `retrievers.py`
One interface, `rank(query, k) -> [(doc_id, score)]`:

- **BM25** — `rank_bm25.BM25Okapi`, lower-cased alphanumeric tokens
- **Dense** — `sentence-transformers/all-MiniLM-L6-v2`; corpus embedded once,
  normalised, ranked by dot product. Chosen because it runs on a laptop CPU in
  minutes; it is *not* a state-of-the-art embedder, and the results should be
  read as "what a small, cheap encoder buys you".
- **Hybrid** — reciprocal rank fusion of the two top-100 lists,
  `score = Σ w / (60 + rank)`, equal weights. RRF needs no score
  normalisation, which is why it is the default choice for fusing systems whose
  scores live on different scales.

### 4. Per-query analysis — `run.py::per_query_analysis`
For each dataset: wins/ties/losses of hybrid against BM25, the queries hybrid
improved most, the queries it hurt most, and queries every retriever scored zero on.

## Results

### Hybrid vs BM25, per query

| Dataset | Hybrid wins | Ties | Hybrid loses | Dense alone beats BM25 |
|---|---|---|---|---|
| SciFact | 70 | 206 | 24 | 70 of 300 |
| NFCorpus | 123 | 145 | 55 | 117 of 323 |

### Where dense wins — paraphrase over exact words

SciFact claims that share no vocabulary with the supporting abstract:

| Query | BM25 | Dense | Hybrid |
|---|---|---|---|
| *Activator-inhibitor pairs are provided dorsally by Admp/chordin.* | 0.33 | 1.00 | 1.00 |
| *Bone marrow cells contribute to adult macrophage compartments.* | 0.36 | 1.00 | 1.00 |

NFCorpus queries that are article titles, not questions:

| Query | BM25 | Dense | Hybrid |
|---|---|---|---|
| *Vitamin C-Enriched Bacon* | 0.00 | 0.63 | 0.63 |
| *Too Much Iodine Can Be as Bad as Too Little* | 0.00 | 0.55 | 0.55 |

### Where dense fails — rare exact terms

| Query | BM25 | Dense | Hybrid |
|---|---|---|---|
| *Ivermectin is used to treat onchocerciasis.* | 1.00 | 0.00 | 0.32 |
| *Venules have a thinner or absent smooth layer compared to arterioles.* | 1.00 | 0.00 | 0.36 |
| *kohlrabi* | 1.00 | 0.00 | 0.39 |
| *Harvard Physicians' Study II* | 0.46 | 0.00 | 0.00 |

A 22M-parameter encoder has no reliable representation of *onchocerciasis* or
*kohlrabi*; BM25 matches the string and is perfect. This is the single most
important finding for anyone building RAG over technical or medical documents:
**drug names, gene symbols, trial acronyms and rare nouns are lexical
territory.** Hybrid recovers part of the loss (0.32–0.39 instead of 0) but
not all of it — equal-weight RRF lets the dense list's noise dilute BM25's
correct top hit.

### What this means for a RAG system

- **Use hybrid by default** on technical corpora; it won on both datasets and
  on the aggregate never lost to either component.
- **Tune the fusion weight per corpus.** Equal weights cost SciFact 24 queries
  that BM25 alone had right. A lexical-leaning weight, or a rule that keeps a
  BM25 exact-match hit in the top-3, would recover most of them.
- **Recall@100 is the number that matters for RAG**, not nDCG@10 — a reranker
  or the LLM can fix ordering, but nothing can use a passage that was never
  retrieved. Hybrid lifts SciFact recall from 0.876 to **0.957**.
- **Log the misses.** `analysis.json` is the artefact a team should review
  weekly; the unanswerable queries are the ones that need corpus fixes, not
  model fixes.

## Reproduce

```bash
pip install -r requirements.txt
# data: BEIR/scifact and BEIR/nfcorpus from Hugging Face -> data/*.parquet (see data.py)
python -m pytest -q                                   # metric + tokenizer tests
PYTHONPATH=src python -m retrieval_eval.run           # both datasets, ~11 min CPU
```

## Layout

```
config.yaml                 datasets, BM25 params, encoder, RRF k and weight, eval k's
src/retrieval_eval/
  data.py                   BEIR loading, graded qrels
  metrics.py                nDCG@k, Recall@k from the definitions
  retrievers.py             BM25, Dense, HybridRRF — one interface
  run.py                    evaluate, per-query analysis, reports/
tests/test_metrics.py
reports/summary.csv · analysis.json · results.json
```

## Design decisions worth defending

- **Evaluate retrieval on its own.** Coupling it to an LLM hides where the
  error came from.
- **Public benchmark with graded judgements**, so numbers are comparable to
  published systems and the metric can be validated.
- **Own metric code, validated.** A retrieval number nobody can check is a
  claim, not a measurement.
- **Per-query analysis over aggregate scores.** The aggregate says hybrid wins;
  the per-query view says *why*, and what to change.
- **A small encoder, deliberately.** The interesting question is not "does a
  huge embedder beat BM25" but "what does the cheap default cost, and where".

## Next

1. Fusion-weight sweep per dataset; exact-match guard for BM25 top hits.
2. A stronger encoder (bge-small / e5-small) as a second dense row.
3. Cross-encoder reranking of the hybrid top-100 — measure nDCG gain vs latency.
4. Connect to the LLM half (groundedness, relevance, coherence judged against
   human labels) — that repo exists as a scaffold and needs an API key.
