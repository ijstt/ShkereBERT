"""Сравнение вариантов retrieval (пул B): dense / +BM25(RRF) / +reranker / всё вместе.

В отличие от eval_e2e (который батчит поиск напрямую через FAISS), здесь каждый вопрос
идёт через продуктовый `DenseRetriever.retrieve()` — иначе reranker и BM25-слияние
не задействуются. Поэтому же отдельно меряем латентность самого поиска.

Для каждого варианта: retrieval-метрики (gold-doc recall@k, MRR) + end-to-end EM/F1
по протоколу calibration/test (tau калибруется на calibration каждого варианта отдельно —
переоценка меняет состав top-k, а значит и распределение gap'ов).

Результаты -> eval/results/retrieval_variants.csv
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace

import pandas as pd

from shkerebert.config import RESULTS_DIR, default_config
from shkerebert.reader import Reader
from shkerebert.retriever import DenseRetriever
from eval.build_corpus import build_eval_set
from eval.eval_e2e import calibrate, score_fixed_tau


def reranker_available(model: str) -> bool:
    import os

    p = os.path.expanduser("~/.cache/huggingface/hub/models--" + model.replace("/", "--"))
    return os.path.isdir(p) and any(True for _ in os.scandir(p))


def collect_via_retrieve(cfg, questions, retriever: DenseRetriever, reader: Reader):
    """Сырые предсказания через retrieve() (учитывает reranker/BM25) + латентности."""
    raw, t_retr = [], 0.0
    t0 = time.time()
    for q in questions:
        t1 = time.time()
        hits = retriever.retrieve(q["question"], k=cfg.retriever.top_k)
        t_retr += time.time() - t1
        top1 = hits[0].dense_score if hits else float("-inf")
        spans = reader.read(q["question"], [h.chunk for h in hits]) if hits else []
        if spans:
            best = spans[0]
            raw.append({"id": q["id"], "text": best.text, "gap": best.gap,
                        "top1_score": top1, "answers": q["answers"],
                        "is_impossible": q["is_impossible"],
                        "hit_doc_ids": [h.chunk.doc_id for h in hits]})
        else:
            raw.append({"id": q["id"], "text": "", "gap": 1e9, "top1_score": top1,
                        "answers": q["answers"], "is_impossible": q["is_impossible"],
                        "hit_doc_ids": []})
    total_ms = 1000 * (time.time() - t0) / max(1, len(questions))
    retr_ms = 1000 * t_retr / max(1, len(questions))
    return raw, total_ms, retr_ms


def retrieval_metrics(raw, questions, k: int):
    """gold-doc recall@k и MRR по answerable-вопросам (gold = context_id)."""
    gold = {q["id"]: q["context_id"] for q in questions}
    answerable = [r for r in raw if not r["is_impossible"]]
    n_hit, rr = 0, 0.0
    for r in answerable:
        ids = r["hit_doc_ids"][:k]
        if gold[r["id"]] in ids:
            n_hit += 1
            rr += 1.0 / (ids.index(gold[r["id"]]) + 1)
    n = max(1, len(answerable))
    return n_hit / n, rr / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800, help="всего вопросов (50/50 calib/test)")
    ap.add_argument("--calib-frac", type=float, default=0.5)
    args = ap.parse_args()

    base = default_config()
    docs, questions = build_eval_set(n_questions=args.n, seed=base.seed)
    n_cal = int(len(questions) * args.calib_frac)
    calib_q, test_q = questions[:n_cal], questions[n_cal:]
    reader = Reader(base.reader)

    variants = [("dense", dict()),
                ("dense+bm25", dict(use_bm25=True))]
    if reranker_available(base.retriever.rerank_model):
        variants += [("dense+rerank", dict(use_reranker=True)),
                     ("dense+bm25+rerank", dict(use_bm25=True, use_reranker=True))]
    else:
        print("! reranker-модель не скачана — варианты с reranker пропущены")

    print(f"Корпус {len(docs)} док., calibration={len(calib_q)}, test={len(test_q)}. "
          f"Строю индекс один раз...")
    retriever = DenseRetriever.build(docs, base, show_progress=False)

    rows = []
    for name, overrides in variants:
        cfg = replace(base, retriever=replace(base.retriever, **overrides))
        retriever.cfg = cfg  # индекс общий, меняется только режим поиска
        print(f"[{name}] прогон...")
        raw_cal, _, _ = collect_via_retrieve(cfg, calib_q, retriever, reader)
        raw_test, total_ms, retr_ms = collect_via_retrieve(cfg, test_q, retriever, reader)
        _, best = calibrate(cfg, raw_cal)
        tau = float(best["tau"])
        res = score_fixed_tau(cfg, raw_test, tau)
        rec, mrr = retrieval_metrics(raw_test, test_q, cfg.retriever.top_k)
        row = {
            "variant": name,
            "recall@k": round(rec, 4), "mrr": round(mrr, 4),
            "EM": round(res["exact"], 2), "F1": round(res["f1"], 2),
            "HasAns_F1": round(res.get("HasAns_f1", float("nan")), 2),
            "NoAns_F1": round(res.get("NoAns_f1", float("nan")), 2),
            "tau*": round(tau, 3),
            "retrieval_ms/q": round(retr_ms, 1),
            "total_ms/q": round(total_ms, 1),
        }
        print("  ", row)
        rows.append(row)

    df = pd.DataFrame(rows)
    df["top_k"] = base.retriever.top_k
    df["chunk_size"] = base.chunk.size
    df["n_test"] = len(test_q)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "retrieval_variants.csv", index=False)
    print("\n", df.to_string(index=False))
    print(f"\nСохранено: {RESULTS_DIR / 'retrieval_variants.csv'}")


if __name__ == "__main__":
    main()
