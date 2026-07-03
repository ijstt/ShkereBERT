"""Multi-seed оценка: доверительные интервалы для EM/F1 (пул A).

Одна цифра F1 на одной выборке — это точечная оценка без представления о разбросе.
Здесь мы повторяем полный протокол (свежая подвыборка вопросов -> свой корпус ->
калибровка tau на calibration -> отчёт на test) на нескольких сидax и отчитываем
mean ± std и 95% CI (t-распределение, малое число прогонов).

Важно: варьируется ИМЕННО выборка данных (корпус+вопросы+сплит), модели детерминированы
на CPU — то есть CI отражает чувствительность метрик к составу выборки.

Результаты -> eval/results/multiseed.csv (+ строка summary в stdout).
"""

from __future__ import annotations

import argparse
import math

import pandas as pd

from shkerebert.config import RESULTS_DIR, default_config
from shkerebert.reader import Reader
from shkerebert.retriever import DenseRetriever
from eval.build_corpus import build_eval_set
from eval.eval_e2e import calibrate, collect_raw_predictions, score_fixed_tau

# Квантили t-распределения (двусторонние 95%) для df = n_seeds - 1.
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365}


def run_seed(seed: int, n: int, calib_frac: float, reader: Reader) -> dict:
    cfg = default_config()
    docs, questions = build_eval_set(n_questions=n, seed=seed)
    n_cal = int(len(questions) * calib_frac)
    calib_q, test_q = questions[:n_cal], questions[n_cal:]
    retriever = DenseRetriever.build(docs, cfg, show_progress=False)

    raw_cal, _ = collect_raw_predictions(cfg, calib_q, retriever, reader)
    raw_test, lat = collect_raw_predictions(cfg, test_q, retriever, reader)
    _, best = calibrate(cfg, raw_cal)
    tau = float(best["tau"])
    res = score_fixed_tau(cfg, raw_test, tau)
    return {
        "seed": seed,
        "tau*": round(tau, 3),
        "EM": round(res["exact"], 2),
        "F1": round(res["f1"], 2),
        "HasAns_F1": round(res.get("HasAns_f1", float("nan")), 2),
        "NoAns_F1": round(res.get("NoAns_f1", float("nan")), 2),
        "latency_ms/q": round(lat, 1),
        "n_test": len(test_q),
    }


def ci95(values) -> tuple[float, float]:
    """(mean, полуширина 95% CI) по t-распределению."""
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, float("nan")
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    t = _T95.get(n - 1, 1.96)
    return mean, t * math.sqrt(var / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--n", type=int, default=800, help="вопросов на сид (50/50 calib/test)")
    ap.add_argument("--calib-frac", type=float, default=0.5)
    args = ap.parse_args()

    reader = Reader(default_config().reader)
    rows = []
    for seed in args.seeds:
        print(f"[seed={seed}] строю корпус и индекс...")
        row = run_seed(seed, args.n, args.calib_frac, reader)
        print("  ", row)
        rows.append(row)

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "multiseed.csv", index=False)

    print("\n=== Multi-seed (n_seeds={}, по {} вопросов) ===".format(len(rows), args.n))
    for col in ["EM", "F1", "HasAns_F1", "NoAns_F1"]:
        mean, half = ci95(list(df[col]))
        print(f"{col:>10}: {mean:.2f} ± {half:.2f} (95% CI), "
              f"min={df[col].min():.2f} max={df[col].max():.2f}")
    print(f"tau* по сидам: {sorted(df['tau*'])}")
    print(f"Сохранено: {RESULTS_DIR / 'multiseed.csv'}")


if __name__ == "__main__":
    main()
