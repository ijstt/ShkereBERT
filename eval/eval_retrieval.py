"""Оценка качества поиска: Recall@k и MRR@k.

Метрики (по answerable-вопросам — для unanswerable нет «правильного» контекста):
  * Recall@k — доля вопросов, у которых gold-контекст попал в top-k найденных чанков;
  * MRR@k    — средний обратный ранг первого gold-чанка.
Помогает обоснованно выбрать эмбеддер и число фрагментов k (шаги 4–5 задания).
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from shkerebert.config import Config, RetrieverConfig, default_config
from shkerebert.retriever import DenseRetriever
from eval.build_corpus import build_eval_set


def evaluate_retrieval(cfg: Config, questions, retriever: DenseRetriever, ks=(1, 3, 5, 10)):
    max_k = max(ks)
    answerable = [q for q in questions if not q["is_impossible"]]

    recall_hits = {k: 0 for k in ks}
    rr_sum = 0.0
    t0 = time.time()
    # Батч-кодируем все вопросы разом (на CPU это на порядок быстрее, чем по одному).
    qvecs = retriever.embedder.encode([q["question"] for q in answerable])
    all_hits = retriever.index.search(qvecs, k=max_k)
    for q, hits in zip(answerable, all_hits):
        gold = q["context_id"]
        ranks = [i for i, h in enumerate(hits) if h.chunk.doc_id == gold]
        first = ranks[0] if ranks else None
        for k in ks:
            if first is not None and first < k:
                recall_hits[k] += 1
        if first is not None:
            rr_sum += 1.0 / (first + 1)
    dt = time.time() - t0

    n = len(answerable)
    rows = {f"recall@{k}": recall_hits[k] / n for k in ks}
    rows["mrr"] = rr_sum / n
    rows["n_answerable"] = n
    rows["latency_ms/q"] = 1000 * dt / max(1, n)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000, help="сколько вопросов брать")
    ap.add_argument("--embed-model", default=None, help="переопределить эмбеддер")
    args = ap.parse_args()

    cfg = default_config()
    if args.embed_model:
        cfg = Config(
            chunk=cfg.chunk,
            retriever=RetrieverConfig(embed_model=args.embed_model, top_k=cfg.retriever.top_k),
            reader=cfg.reader,
        )

    docs, questions = build_eval_set(n_questions=args.n, seed=cfg.seed)
    print(f"Корпус: {len(docs)} документов, {len(questions)} вопросов. Строю индекс...")
    retriever = DenseRetriever.build(docs, cfg)

    res = evaluate_retrieval(cfg, questions, retriever)
    df = pd.DataFrame([{"embed_model": cfg.retriever.embed_model,
                        "chunk_size": cfg.chunk.size, **res}])
    from shkerebert.config import RESULTS_DIR
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "retrieval.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nСохранено: {out}")


if __name__ == "__main__":
    main()
