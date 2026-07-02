"""Сквозная (end-to-end) оценка QA: EM / F1 по официальной метрике squad_v2.

Ключевые идеи:
  * прогоняем полный пайплайн (retrieval -> reader) один раз, СОХРАНЯЯ «сырые» предсказания:
    лучший span-ответ и его gap = null_score - best_span_score;
  * затем СВИПАЕМ порог абстенции tau по сетке, не перезапуская модель: для каждого tau
    решаем «нет ответа», если gap > tau, и считаем EM/F1 (overall + HasAns + NoAns);
  * выбираем tau*, максимизирующий overall F1, строим график F1 vs tau (PNG) и таблицу.

Так метрики честные (официальный squad_v2), а калибровка порога прозрачна и воспроизводима.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from shkerebert.config import Config, RESULTS_DIR, default_config
from shkerebert.reader import Reader
from shkerebert.retriever import DenseRetriever
from eval.build_corpus import build_eval_set


def collect_raw_predictions(cfg: Config, questions, retriever: DenseRetriever, reader: Reader):
    """Один прогон: для каждого вопроса — лучший ответ, его gap и retrieval-скор top1."""
    raw = []
    t0 = time.time()
    # Батч-кодируем вопросы и ищем разом; reader (дорогой) — уже поштучно.
    qvecs = retriever.embedder.encode([q["question"] for q in questions])
    all_hits = retriever.index.search(qvecs, k=cfg.retriever.top_k)
    for q, hits in zip(questions, all_hits):
        top1 = hits[0].score if hits else float("-inf")
        spans = reader.read(q["question"], [h.chunk for h in hits]) if hits else []
        if spans:
            best = spans[0]
            raw.append({
                "id": q["id"],
                "text": best.text,
                "gap": best.gap,              # >0 => склонность к «нет ответа»
                "top1_score": top1,
                "answers": q["answers"],
                "is_impossible": q["is_impossible"],
            })
        else:
            raw.append({
                "id": q["id"], "text": "", "gap": 1e9, "top1_score": top1,
                "answers": q["answers"], "is_impossible": q["is_impossible"],
            })
    dt = time.time() - t0
    return raw, 1000 * dt / max(1, len(questions))


def _references(raw):
    refs = []
    for r in raw:
        texts = r["answers"]
        refs.append({
            "id": r["id"],
            "answers": {"text": texts, "answer_start": [0] * len(texts)},
        })
    return refs


def score_at_tau(metric, raw, refs, tau: float, min_score: float):
    preds = []
    for r in raw:
        abstain = (r["gap"] > tau) or (r["top1_score"] < min_score) or (not r["text"])
        preds.append({
            "id": r["id"],
            "prediction_text": "" if abstain else r["text"],
            "no_answer_probability": 1.0 if abstain else 0.0,
        })
    res = metric.compute(predictions=preds, references=refs, no_answer_threshold=0.5)
    return res


def calibrate(cfg: Config, raw, taus=None):
    import evaluate

    metric = evaluate.load("squad_v2")
    refs = _references(raw)
    gaps = np.array([r["gap"] for r in raw if np.isfinite(r["gap"])])
    if taus is None:
        lo, hi = np.percentile(gaps, 2), np.percentile(gaps, 98)
        taus = np.linspace(lo, hi, 31)

    curve = []
    for tau in taus:
        res = score_at_tau(metric, raw, refs, float(tau), cfg.retriever.min_score)
        curve.append({
            "tau": float(tau),
            "exact": res["exact"], "f1": res["f1"],
            "HasAns_f1": res.get("HasAns_f1", float("nan")),
            "NoAns_f1": res.get("NoAns_f1", float("nan")),
        })
    curve_df = pd.DataFrame(curve)
    best = curve_df.loc[curve_df["f1"].idxmax()]
    return curve_df, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    args = ap.parse_args()

    cfg = default_config()
    docs, questions = build_eval_set(n_questions=args.n, seed=cfg.seed)
    print(f"Корпус: {len(docs)} документов, {len(questions)} вопросов. Строю пайплайн...")
    retriever = DenseRetriever.build(docs, cfg)
    reader = Reader(cfg.reader)

    raw, lat = collect_raw_predictions(cfg, questions, retriever, reader)
    print(f"Инференс: {lat:.1f} ms/вопрос. Калибрую порог tau...")
    curve_df, best = calibrate(cfg, raw)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    curve_df.to_csv(RESULTS_DIR / "e2e_threshold_curve.csv", index=False)

    # График F1 vs tau
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4.5))
    plt.plot(curve_df["tau"], curve_df["f1"], label="overall F1", lw=2)
    plt.plot(curve_df["tau"], curve_df["HasAns_f1"], "--", label="HasAns F1")
    plt.plot(curve_df["tau"], curve_df["NoAns_f1"], "--", label="NoAns F1")
    plt.axvline(best["tau"], color="red", ls=":", label=f"tau*={best['tau']:.2f}")
    plt.xlabel("порог абстенции tau (null - span)")
    plt.ylabel("F1")
    plt.title("Калибровка порога «нет ответа» (SQuAD v2)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "e2e_threshold_curve.png", dpi=130)

    summary = pd.DataFrame([{
        "tau*": best["tau"], "EM": best["exact"], "F1": best["f1"],
        "HasAns_F1": best["HasAns_f1"], "NoAns_F1": best["NoAns_f1"],
        "latency_ms/q": lat, "n": len(questions),
        "embed_model": cfg.retriever.embed_model, "reader": cfg.reader.model,
        "chunk_size": cfg.chunk.size, "top_k": cfg.retriever.top_k,
    }])
    summary.to_csv(RESULTS_DIR / "e2e_summary.csv", index=False)
    print("\n=== Лучшая рабочая точка ===")
    print(summary.to_string(index=False))
    print(f"\nГрафик: {RESULTS_DIR/'e2e_threshold_curve.png'}")


if __name__ == "__main__":
    main()
