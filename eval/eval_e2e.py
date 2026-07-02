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


def collect_oracle_predictions(cfg: Config, questions, reader: Reader):
    """Oracle-baseline: reader читает ЗОЛОТОЙ контекст вопроса (без retrieval).

    Показывает потолок reader'а — если end-to-end F1 близок к нему, значит retrieval
    почти не теряет качества (иначе виден размер потери на этапе поиска).
    """
    from shkerebert.chunking import Chunk

    raw, t0 = [], time.time()
    for q in questions:
        gold = Chunk(id=f"{q['context_id']}::gold", doc_id=q["context_id"],
                     text=q["context"], index=0, n_tokens=0, title="")
        spans = reader.read(q["question"], [gold])
        best = spans[0]
        raw.append({
            "id": q["id"], "text": best.text, "gap": best.gap,
            "top1_score": 1e9,  # золотой контекст всегда «найден»
            "answers": q["answers"], "is_impossible": q["is_impossible"],
        })
    dt = time.time() - t0
    return raw, 1000 * dt / max(1, len(questions))


def score_fixed_tau(cfg: Config, raw, tau: float):
    """Метрики на выборке при ФИКСИРОВАННОМ tau (для отчёта на held-out test)."""
    import evaluate

    metric = evaluate.load("squad_v2")
    refs = _references(raw)
    return score_at_tau(metric, raw, refs, tau, cfg.retriever.min_score)


def _summary_row(setup, tau, res, lat, n):
    return {
        "setup": setup, "tau*": round(float(tau), 3),
        "EM": round(res["exact"], 2), "F1": round(res["f1"], 2),
        "HasAns_F1": round(res.get("HasAns_f1", float("nan")), 2),
        "NoAns_F1": round(res.get("NoAns_f1", float("nan")), 2),
        "latency_ms/q": round(lat, 1), "n_test": n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500, help="всего вопросов (делятся 50/50)")
    ap.add_argument("--calib-frac", type=float, default=0.5)
    args = ap.parse_args()

    cfg = default_config()
    docs, questions = build_eval_set(n_questions=args.n, seed=cfg.seed)
    n_cal = int(len(questions) * args.calib_frac)
    calib_q, test_q = questions[:n_cal], questions[n_cal:]
    print(f"Корпус: {len(docs)} док.; calibration={len(calib_q)}, test={len(test_q)}. "
          f"Строю пайплайн...")
    retriever = DenseRetriever.build(docs, cfg)
    reader = Reader(cfg.reader)

    # --- Retrieval-пайплайн: калибруем tau на calibration, отчитываемся на TEST ---
    raw_cal_r, _ = collect_raw_predictions(cfg, calib_q, retriever, reader)
    raw_test_r, lat_r = collect_raw_predictions(cfg, test_q, retriever, reader)
    curve_r, best_r = calibrate(cfg, raw_cal_r)
    tau_r = float(best_r["tau"])
    test_r = score_fixed_tau(cfg, raw_test_r, tau_r)
    print(f"[retrieval] tau*={tau_r:.2f} (на calibration) -> оценка на test")

    # --- Oracle-baseline (reader на золотом контексте) ---
    raw_cal_o, _ = collect_oracle_predictions(cfg, calib_q, reader)
    raw_test_o, lat_o = collect_oracle_predictions(cfg, test_q, reader)
    _, best_o = calibrate(cfg, raw_cal_o)
    tau_o = float(best_o["tau"])
    test_o = score_fixed_tau(cfg, raw_test_o, tau_o)
    print(f"[oracle]    tau*={tau_o:.2f} (на calibration) -> оценка на test")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    curve_r.to_csv(RESULTS_DIR / "e2e_threshold_curve.csv", index=False)

    # График F1 vs tau (калибровка retrieval-пайплайна, отметка выбранного tau*)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4.5))
    plt.plot(curve_r["tau"], curve_r["f1"], label="overall F1", lw=2)
    plt.plot(curve_r["tau"], curve_r["HasAns_f1"], "--", label="HasAns F1")
    plt.plot(curve_r["tau"], curve_r["NoAns_f1"], "--", label="NoAns F1")
    plt.axvline(tau_r, color="red", ls=":", label=f"tau*={tau_r:.2f} (calib)")
    plt.xlabel("порог абстенции tau (null - span)")
    plt.ylabel("F1")
    plt.title("Калибровка порога на calibration-сплите (SQuAD v2)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "e2e_threshold_curve.png", dpi=130)

    summary = pd.DataFrame([
        _summary_row("retrieval (test)", tau_r, test_r, lat_r, len(test_q)),
        _summary_row("oracle-context (test)", tau_o, test_o, lat_o, len(test_q)),
    ])
    summary["embed_model"] = cfg.retriever.embed_model
    summary["reader"] = cfg.reader.model
    summary["chunk_size"] = cfg.chunk.size
    summary["top_k"] = cfg.retriever.top_k
    summary.to_csv(RESULTS_DIR / "e2e_summary.csv", index=False)

    gap = test_o["f1"] - test_r["f1"]
    print("\n=== Оценка на HELD-OUT TEST (tau подобран на calibration) ===")
    print(summary[["setup", "tau*", "EM", "F1", "HasAns_F1", "NoAns_F1", "latency_ms/q", "n_test"]]
          .to_string(index=False))
    print(f"\nПотеря на этапе retrieval (oracle F1 - retrieval F1): {gap:.2f} пункта F1")
    print(f"График: {RESULTS_DIR/'e2e_threshold_curve.png'}")


if __name__ == "__main__":
    main()
