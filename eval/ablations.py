"""Ablation-исследования — ядро доказательства эффективности.

Сравниваем варианты пайплайна на одном и том же оценочном наборе и показываем, какой
выбор параметров лучше и почему:
  * размер чанка (chunk_size) — влияет на Recall и F1 (обоснование шага 2 задания);
  * число фрагментов top_k — покрытие vs шум;
  * reranker on/off — вклад кросс-энкодера (stretch).

Для каждой конфигурации считаем retrieval Recall@k/MRR и end-to-end F1 (с калибровкой tau).
Результаты -> CSV + PNG в eval/results/. Индекс переэмбеддится только при смене чанкинга
(эмбеддинги кэшируются), поэтому свип по top_k/reranker дешёвый.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import pandas as pd

from shkerebert.config import Config, RESULTS_DIR, RetrieverConfig, default_config
from shkerebert.reader import Reader
from shkerebert.retriever import DenseRetriever
from eval.build_corpus import build_eval_set
from eval.eval_retrieval import evaluate_retrieval
from eval.eval_e2e import collect_raw_predictions, calibrate


def run_config(cfg: Config, questions, retriever, reader):
    k = cfg.retriever.top_k
    # dedup ks, иначе при top_k==1 recall@1 считается дважды.
    ks = tuple(sorted({1, k}))
    retr = evaluate_retrieval(cfg, questions, retriever, ks=ks)
    raw, lat = collect_raw_predictions(cfg, questions, retriever, reader)
    _, best = calibrate(cfg, raw)
    return {
        "chunk_size": cfg.chunk.size,
        "strategy": cfg.chunk.strategy,
        "top_k": k,
        "reranker": cfg.retriever.use_reranker,
        "recall@1": retr["recall@1"],
        "recall@k": retr[f"recall@{k}"],   # recall при этом top_k
        "mrr": retr["mrr"],
        "EM": best["exact"],
        "F1": best["f1"],
        "NoAns_F1": best["NoAns_f1"],
        "tau*": best["tau"],
        "latency_ms/q": lat,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500, help="вопросов на конфигурацию")
    ap.add_argument("--chunk-sizes", type=int, nargs="+", default=[128, 256, 512])
    ap.add_argument("--top-ks", type=int, nargs="+", default=[1, 3, 5, 10])
    ap.add_argument("--reranker", action="store_true", help="добавить прогон с reranker")
    args = ap.parse_args()

    base = default_config()
    docs, questions = build_eval_set(n_questions=args.n, seed=base.seed)
    reader = Reader(base.reader)

    rows = []
    for size in args.chunk_sizes:
        cfg_chunk = replace(base, chunk=replace(base.chunk, size=size))
        print(f"[chunk_size={size}] строю индекс...")
        retriever = DenseRetriever.build(docs, cfg_chunk, show_progress=False)
        for k in args.top_ks:
            variants = [False, True] if (args.reranker and k == max(args.top_ks)) else [False]
            for use_rr in variants:
                cfg = replace(
                    cfg_chunk,
                    retriever=RetrieverConfig(
                        embed_model=base.retriever.embed_model,
                        top_k=k, use_reranker=use_rr,
                    ),
                )
                retriever.cfg = cfg  # переиспользуем индекс, меняем только режим поиска
                row = run_config(cfg, questions, retriever, reader)
                print("  ", row)
                rows.append(row)

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "ablations.csv", index=False)

    _plot(df)
    print("\n", df.to_string(index=False))
    print(f"\nСохранено: {RESULTS_DIR/'ablations.csv'} и PNG-графики")


def _plot(df: pd.DataFrame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dense = df[~df["reranker"]]
    # F1 vs top_k по размерам чанка
    plt.figure(figsize=(7, 4.5))
    for size, g in dense.groupby("chunk_size"):
        g = g.sort_values("top_k")
        plt.plot(g["top_k"], g["F1"], marker="o", label=f"chunk={size}")
    plt.xlabel("top_k"); plt.ylabel("end-to-end F1")
    plt.title("F1 vs top_k при разных размерах чанка")
    plt.legend(); plt.tight_layout()
    plt.savefig(RESULTS_DIR / "ablation_f1_topk.png", dpi=130)

    # Recall vs top_k
    plt.figure(figsize=(7, 4.5))
    for size, g in dense.groupby("chunk_size"):
        g = g.sort_values("top_k")
        plt.plot(g["top_k"], g["recall@k"], marker="s", label=f"chunk={size}")
    plt.xlabel("top_k"); plt.ylabel("Recall@top_k")
    plt.title("Recall vs top_k при разных размерах чанка")
    plt.legend(); plt.tight_layout()
    plt.savefig(RESULTS_DIR / "ablation_recall_topk.png", dpi=130)


if __name__ == "__main__":
    main()
