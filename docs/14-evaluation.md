# 14. Оценка качества

Каталог `eval/` — воспроизводимые эксперименты. Результаты → `eval/results/`.

## Общий протокол

1. **Корпус** — `build_eval_set(n, seed=42)`: случайная подвыборка вопросов dev SQuAD v2 +
   уникальные их контексты как документы.
2. **Calibration/test split** — первые 50% вопросов → калибровка τ, вторые 50% → отчёт
   (без утечки).
3. **Метрики** — HuggingFace `evaluate.load("squad_v2")`: EM, F1, HasAns_f1, NoAns_f1.

---

## build_corpus.py

```19:44:eval/build_corpus.py
def build_eval_set(split="validation", n_questions=2000, seed=42):
    all_q = load_squad_questions(split=split)
    rng = random.Random(seed)
    rng.shuffle(all_q)
    questions = all_q[:min(n_questions, len(all_q))]
    # docs = unique contexts of these questions
```

**qrel:** gold-документ = `context_id` вопроса.

---

## eval_retrieval.py — Recall@k, MRR

```21:47:eval/eval_retrieval.py
def evaluate_retrieval(cfg, questions, retriever, ks=(1, 3, 5, 10)):
    answerable = [q for q in questions if not q["is_impossible"]]
    qvecs = retriever.embedder.encode([q["question"] for q in answerable])
    all_hits = retriever.index.search(qvecs, k=max_k)
    for q, hits in zip(answerable, all_hits):
        gold = q["context_id"]
        ranks = [i for i, h in enumerate(hits) if h.chunk.doc_id == gold]
```

Запуск:

```bash
.venv/bin/python -m eval.eval_retrieval --n 1000
.venv/bin/python -m eval.eval_retrieval --n 1000 --embed-model sentence-transformers/...
```

Выход: `eval/results/retrieval.csv`.

---

## eval_e2e.py — end-to-end EM/F1 + калибровка τ

### collect_raw_predictions

Один прогон retrieval + reader, сохраняет `gap`, `text`, `top1_score` без решения
об abstention:

```27:53:eval/eval_e2e.py
def collect_raw_predictions(cfg, questions, retriever, reader):
    qvecs = retriever.embedder.encode([q["question"] for q in questions])
    all_hits = retriever.index.search(qvecs, k=cfg.retriever.top_k)
    for q, hits in zip(questions, all_hits):
        spans = reader.read(q["question"], [h.chunk for h in hits])
        raw.append({"gap": best.gap, "text": best.text, "top1_score": top1, ...})
```

### calibrate — свип τ без перезапуска модели

```80:101:eval/eval_e2e.py
def calibrate(cfg, raw, taus=None):
    gaps = np.array([r["gap"] for r in raw if np.isfinite(r["gap"])])
    taus = np.linspace(np.percentile(gaps, 2), np.percentile(gaps, 98), 31)
    for tau in taus:
        res = score_at_tau(metric, raw, refs, tau, cfg.retriever.min_score)
    best = curve_df.loc[curve_df["f1"].idxmax()]
```

### Oracle baseline

Reader на золотом контексте — верхняя граница без ошибок retrieval:

```104:124:eval/eval_e2e.py
def collect_oracle_predictions(cfg, questions, reader):
    gold = Chunk(id=f"{q['context_id']}::gold", text=q["context"], ...)
    spans = reader.read(q["question"], [gold])
```

Выход:

- `e2e_summary.csv` — retrieval vs oracle на test;
- `e2e_threshold_curve.csv` + `.png` — F1 vs τ.

Запуск: `python -m eval.eval_e2e --n 1500 --calib-frac 0.5`

---

## ablations.py — chunk_size × top_k

```52:82:eval/ablations.py
for size in args.chunk_sizes:  # [128, 256, 512]
    retriever = DenseRetriever.build(docs, cfg_chunk)
    for k in args.top_ks:      # [1, 3, 5, 10]
        row = run_config(cfg, questions, retriever, reader)
```

На каждую конфигурацию: Recall@1, Recall@k, EM, F1, τ*, latency.

Индекс переэмбеддивается только при смене chunk_size (кэш embeddings).

Выход: `ablations.csv`, `ablation_f1_topk.png`, `ablation_recall_topk.png`.

---

## eval_longdoc.py — длинные документы

Склеивает все абзацы одной Wikipedia-статьи (по `title`) в один документ:

```42:71:eval/eval_longdoc.py
def build_longdoc_eval_set(...):
    for q in all_q:
        if q["title"] in titles and q["context"] not in seen:
            by_title.setdefault(q["title"], []).append(q["context"])
    docs = [Document(text="\n\n".join(ctxs), ...) for title, ctxs in by_title.items()]
```

Метрики:

- **answer-recall@k** — эталонный span содержится в тексте top-k чанка (строже doc-recall);
- **doc-recall@k** — хотя бы один чанк из той же статьи;
- e2e EM/F1 с калибровкой τ.

Overlap = `size // 4` (25%).

Вывод v1 «chunk size не важен» был артефактом коротких SQuAD-контекстов.

---

## eval_multiseed.py — доверительные интервалы

Повторяет полный протокол на seeds `[42, 43, 44, 45, 46]`:

```31:52:eval/eval_multiseed.py
def run_seed(seed, n, calib_frac, reader):
    docs, questions = build_eval_set(n_questions=n, seed=seed)
    ...
    return {"EM", "F1", "HasAns_F1", "NoAns_F1", "tau*", ...}
```

Отчёт: mean ± 95% CI (t-распределение). F1 = 76.9 ± 3.9 — baseline 75.8 внутри CI.

---

## eval_retrieval_variants.py — dense vs BM25 vs reranker

Прогон через **продуктовый** `DenseRetriever.retrieve()` (не прямой FAISS):

```36:58:eval/eval_retrieval_variants.py
def collect_via_retrieve(cfg, questions, retriever, reader):
    for q in questions:
        hits = retriever.retrieve(q["question"], k=cfg.retriever.top_k)
```

Отдельная калибровка τ для каждого variant.

---

## eval_generative.py — сравнение режимов ридера

Через `RAGPipeline.answer()` на seed=123 (не 42 — без утечки τ):

```58:78:eval/eval_generative.py
def run_mode(pipe, questions, mode):
    ans = pipe.answer(q["question"], mode=mode)
```

Метрики:

| Метрика | Описание |
|---------|----------|
| EM / F1 | squad_v2 на cleaned ответе |
| gold_containment | эталон ⊆ ответ (для generative) |
| abstain_recall_noans | доля отказов на impossible |
| false_abstain_hasans | ложные отказы на answerable |
| unverified_flag_noans | UNVERIFIED на impossible (hybrid) |
| latency p50/p95 | задержка |

Выход: `generative_eval.csv`, `generative_eval_details.csv`.

---

## Официальные метрики SQuAD v2

**Exact Match (EM):** нормализованное предсказание == любой эталон.

**F1:** token-level overlap между pred и gold (max по эталонам).

**HasAns_f1 / NoAns_f1:** F1 отдельно на answerable / unanswerable подмножествах.

Abstention: `prediction_text=""` + `no_answer_probability=1.0` при `no_answer_threshold=0.5`.

---

## Быстрый запуск всех eval

```bash
./run.sh eval        # retrieval + e2e (~30–60 мин CPU)
./run.sh eval-extra  # longdoc + multiseed + variants + generative (часы)
```

## RESULTS_DIR

Все eval-скрипты пишут отчёты в константу из конфига:

```python
# shkerebert/config.py
RESULTS_DIR = Path(__file__).resolve().parent.parent / "eval" / "results"
```
