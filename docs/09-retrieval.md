# 09. Retrieval

Модуль `shkerebert/retriever.py`.

## DenseRetriever — основной класс

```45:67:shkerebert/retriever.py
class DenseRetriever:
    def __init__(self, index, embedder, cfg):
        self.index = index
        self.embedder = embedder
        self.cfg = cfg
        self._bm25 = None  # лениво
        self._pos = None   # chunk.id -> позиция

    @classmethod
    def build(cls, docs, cfg, show_progress=True):
        chunks = chunk_documents(docs, cfg.chunk, cfg.retriever.embed_model)
        embeddings, embedder = embed_corpus(chunks, cfg, show_progress=show_progress)
        index = VectorIndex(chunks, embeddings)
        return cls(index, embedder, cfg)
```

## Базовый dense-поиск

```69:94:shkerebert/retriever.py
    def retrieve(self, question, k=None):
        rcfg = self.cfg.retriever
        k = k or rcfg.top_k
        pool = k
        if rcfg.use_reranker:
            pool = max(pool, rcfg.rerank_pool)
        if rcfg.use_bm25:
            pool = max(pool, rcfg.bm25_pool)
        qvec = self.embedder.encode([question])
        hits = self.index.search(qvec, k=pool)[0]
        cands = [RetrievedChunk(h.chunk, h.score, h.score) for h in hits]
        ...
        return cands[:k]
```

**Pool vs k:** если включены reranker или BM25, из FAISS достаётся **больший** pool
(20 или 50), затем переоценка/фьюз, и обрезка до `k`.

`dense_score` = исходный FAISS-скор (косинус). Сохраняется **отдельно** от финального
`score` — на `dense_score` работает retrieval-level абстенция в pipeline.

## BM25 — лексический поиск (stretch)

**BM25 (Best Matching 25)** — классическая формула ранжирования по частоте терминов.
Ловит точные совпадения (номера тарифов, редкие термины), где dense «плывёт».

```54:59:shkerebert/retriever.py
    @property
    def bm25(self):
        if self._bm25 is None:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi([_bm25_tokens(c.text) for c in self.index.chunks])
        return self._bm25
```

Токенизация: `\w+` в lower case (`_WORD = re.compile(r"[\w']+", re.UNICODE)`).

## Reciprocal Rank Fusion (RRF)

Dense и BM25 выдают **несравнимые** шкалы скоров. Фьюзим **ранги**, не скоры:

```
RRF(doc) = Σ  1 / (k + rank_i + 1)
```

где `k = rrf_k` (default 60), `rank_i` — позиция документа в i-м списке.

```96:125:shkerebert/retriever.py
    def _fuse_bm25(self, question, qvec, dense_cands):
        bm_scores = self.bm25.get_scores(_bm25_tokens(question))
        bm_top = [int(i) for i in np.argsort(bm_scores)[::-1][:rcfg.bm25_pool]
                  if bm_scores[i] > 0]
        for rank, c in enumerate(dense_cands):
            rrf[i] += 1.0 / (rcfg.rrf_k + rank + 1)
            dense[i] = c.dense_score
        for rank, i in enumerate(bm_top):
            rrf[i] += 1.0 / (rcfg.rrf_k + rank + 1)
            if i not in dense:
                dense[i] = float(np.dot(qvec, self.index.index.reconstruct(i)))
        order = sorted(rrf, key=rrf.get, reverse=True)
        return [RetrievedChunk(chunks[i], rrf[i], dense[i]) for i in order]
```

**Финальный `score`** после RRF — сумма RRF-вкладов (не косинус).
**`dense_score`** — по-прежнему косинус для абстенции.

Тест: `tests/test_retriever.py::test_bm25_hybrid_finds_exact_term` — тариф `ZX-9917`.

## Cross-Encoder Reranker (stretch)

**Bi-encoder** (sentence-transformers): question и passage кодируются **независимо**.
**Cross-encoder**: пара `(question, passage)` подаётся в один Transformer — точнее, но
O(n) forward pass на каждого кандидата.

```87:92:shkerebert/retriever.py
        if rcfg.use_reranker:
            reranker = _load_reranker(rcfg.rerank_model)
            scores = reranker.predict([(question, c.chunk.text) for c in cands])
            cands = [RetrievedChunk(c.chunk, float(s), c.dense_score)
                     for c, s in zip(cands, scores)]
            cands.sort(key=lambda r: r.score, reverse=True)
```

Модель: `cross-encoder/ms-marco-MiniLM-L-6-v2` (обучена на MS MARCO passage ranking).

Reranker можно комбинировать с BM25: сначала RRF, потом rerank top-pool.

## Метрики retrieval (eval)

| Метрика | Формула | Смысл |
|---------|---------|-------|
| Recall@k | доля вопросов, где gold-doc в top-k | Покрытие |
| MRR | mean 1/(rank+1) первого gold | Качество ранжирования |

Gold = `context_id` вопроса (документ SQuAD). Считается только на **answerable** —
у unanswerable нет «правильного» контекста.

Baseline Recall@5 = **0.942** → обоснование `top_k=5`.

## Эксперимент: варианты retrieval

`eval/eval_retrieval_variants.py` сравнивает:

| variant | recall@5 | e2e F1 (test) |
|---------|----------|---------------|
| dense | 0.955 | 80.11 |
| dense+bm25 | 0.972 | 80.11 |
| dense+rerank | — | — |
| dense+bm25+rerank | — | — |

BM25 улучшает recall/MRR (+0.9 мс), но e2e F1 на SQuAD не меняется — вопросы
перефразированы, точные термины редки. Выигрыш ожидается на доменных доках (банк, тарифы).

## Выбор top_k

Ablation показывает: F1 насыщается к k=3–5, латентность reader растёт линейно
(k=1 ≈ 50 мс → k=10 ≈ 550 мс, т.к. reader читает каждый фрагмент).

**k=5** — компромисс: Recall@5 ≈ 0.95, F1 около пика, умеренная задержка.
