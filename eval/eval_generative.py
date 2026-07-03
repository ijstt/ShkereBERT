"""Оценка генеративного и гибридного режимов против extractive (пул D).

Все три режима гоняются через РЕАЛЬНЫЙ продуктовый путь (RAGPipeline.answer) на одной
и той же свежей подвыборке SQuAD v2 (сид не совпадает с калибровочным, чтобы не было
утечки: tau экстрактивного ридера откалиброван на другой выборке).

Почему EM/F1 «в лоб» несправедливы к генерации: SQuAD-метрика ждёт короткий span, а
LLM отвечает фразой («The transformer was introduced in 2017 [1]») — F1 штрафует
многословие. Поэтому отчитываем ДВА взгляда:
  * строгий squad_v2 EM/F1 по очищенному ответу (сняты цитаты [n], префиксы);
  * gold-containment — доля answerable-вопросов, где нормализованный эталон
    содержится в нормализованном ответе (адекватная метрика для генерации).

Плюс качество абстенции (матрица: abstain x is_impossible) и латентность (mean/p50/p95).

Результаты -> eval/results/generative_eval.csv (summary),
              generative_eval_details.csv (по-вопросные, для анализа ошибок).
"""

from __future__ import annotations

import argparse
import re
import string
import time

import numpy as np
import pandas as pd

from shkerebert.config import RESULTS_DIR, default_config
from shkerebert.pipeline import RAGPipeline
from eval.build_corpus import build_eval_set

_CITE = re.compile(r"\s*\[\d+\]")
_PREFIX = re.compile(r"^(ответ|answer)\s*[:\-—]\s*", re.IGNORECASE)


def clean_generated(text: str) -> str:
    """Снять цитаты [n] и служебные префиксы — оставить сам ответ."""
    text = _CITE.sub("", text)
    text = _PREFIX.sub("", text.strip())
    return text.strip().strip(".")


def _normalize(text: str) -> str:
    """Нормализация в духе squad_v2: lower, без пунктуации/артиклей, схлопнутые пробелы."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def contains_gold(pred: str, golds: list[str]) -> bool:
    p = _normalize(pred)
    return any(_normalize(g) and _normalize(g) in p for g in golds)


def run_mode(pipe: RAGPipeline, questions, mode: str):
    details = []
    for i, q in enumerate(questions):
        t0 = time.time()
        ans = pipe.answer(q["question"], mode=mode)
        dt = time.time() - t0
        details.append({
            "id": q["id"],
            "mode": mode,
            "question": q["question"],
            "gold": " | ".join(q["answers"]),
            "is_impossible": q["is_impossible"],
            "raw_answer": ans.answer,
            "answer": clean_generated(ans.answer) if mode != "extractive" else ans.answer,
            "abstained": not ans.is_answerable,
            "reason": ans.reason,
            "latency_s": round(dt, 3),
        })
        if (i + 1) % 25 == 0:
            print(f"  [{mode}] {i + 1}/{len(questions)}", flush=True)
    return details


def summarize(details, mode: str) -> dict:
    import evaluate

    metric = evaluate.load("squad_v2")
    preds, refs = [], []
    for d in details:
        preds.append({
            "id": d["id"],
            "prediction_text": "" if d["abstained"] else d["answer"],
            "no_answer_probability": 1.0 if d["abstained"] else 0.0,
        })
        golds = [g for g in d["gold"].split(" | ") if g]
        refs.append({"id": d["id"],
                     "answers": {"text": golds, "answer_start": [0] * len(golds)}})
    res = metric.compute(predictions=preds, references=refs, no_answer_threshold=0.5)

    ans_q = [d for d in details if not d["is_impossible"]]
    noans_q = [d for d in details if d["is_impossible"]]
    answered_ans = [d for d in ans_q if not d["abstained"]]
    contain = sum(contains_gold(d["answer"], d["gold"].split(" | ")) for d in answered_ans)
    lat = np.array([d["latency_s"] for d in details])

    return {
        "mode": mode,
        "EM": round(res["exact"], 2),
        "F1": round(res["f1"], 2),
        "HasAns_F1": round(res.get("HasAns_f1", float("nan")), 2),
        "NoAns_F1": round(res.get("NoAns_f1", float("nan")), 2),
        # доля отвеченных answerable, где эталон содержится в ответе
        "gold_containment": round(contain / max(1, len(answered_ans)), 4),
        # абстенция: корректный отказ на impossible / ложный отказ на answerable
        "abstain_recall_noans": round(
            sum(d["abstained"] for d in noans_q) / max(1, len(noans_q)), 4),
        "false_abstain_hasans": round(
            sum(d["abstained"] for d in ans_q) / max(1, len(ans_q)), 4),
        # hybrid-guardrail: какую долю ответов на impossible-вопросы верификатор
        # пометил UNVERIFIED (перехваченные галлюцинации); для остальных режимов 0
        "unverified_flag_noans": round(
            sum("UNVERIFIED" in d["reason"] for d in noans_q if not d["abstained"])
            / max(1, sum(not d["abstained"] for d in noans_q)), 4),
        "latency_mean_s": round(float(lat.mean()), 2),
        "latency_p50_s": round(float(np.percentile(lat, 50)), 2),
        "latency_p95_s": round(float(np.percentile(lat, 95)), 2),
        "n": len(details),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150, help="вопросов (генерация медленная на CPU)")
    ap.add_argument("--seed", type=int, default=123, help="НЕ 42: свежая выборка без утечки tau")
    ap.add_argument("--modes", nargs="+", default=["extractive", "generative", "hybrid"])
    args = ap.parse_args()

    cfg = default_config()
    docs, questions = build_eval_set(n_questions=args.n, seed=args.seed)
    n_imp = sum(q["is_impossible"] for q in questions)
    print(f"Выборка: {len(questions)} вопросов ({n_imp} unanswerable), "
          f"{len(docs)} документов. Строю пайплайн...")
    pipe = RAGPipeline.build(docs, cfg, show_progress=False)

    all_details, rows = [], []
    for mode in args.modes:
        print(f"[{mode}] прогон {len(questions)} вопросов...")
        details = run_mode(pipe, questions, mode)
        all_details.extend(details)
        row = summarize(details, mode)
        print("  ", row)
        rows.append(row)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "generative_eval.csv", index=False)
    pd.DataFrame(all_details).to_csv(RESULTS_DIR / "generative_eval_details.csv", index=False)

    print("\n=== Сравнение режимов (одна выборка, seed={}) ===".format(args.seed))
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nСохранено: {RESULTS_DIR / 'generative_eval.csv'} (+details)")


if __name__ == "__main__":
    main()
