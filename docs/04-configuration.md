# 04. Конфигурация

Все параметры пайплайна собраны в **frozen dataclasses** — один источник правды,
иммутабельность, явная типизация.

Файл: `shkerebert/config.py`.

## Иерархия конфига

```
Config
├── chunk: ChunkConfig
├── retriever: RetrieverConfig
├── reader: ReaderConfig
├── generator: GeneratorConfig
├── seed: int = 42
└── reader_mode: str = "extractive"
```

## ChunkConfig — параметры чанкинга

```18:23:shkerebert/config.py
@dataclass(frozen=True)
class ChunkConfig:
    size: int = 256          # целевой размер чанка в токенах эмбеддера
    overlap: int = 64        # перекрытие соседних чанков (токены)
    strategy: str = "sentence"  # "sentence" | "fixed"
```

- **size** — бюджет в токенах токенизатора эмбеддера (не символах!).
- **overlap** — сколько токенов «переносится» с конца предыдущего чанка в начало
  следующего; защищает от разрезания ответа на границе.
- **strategy**:
  - `"sentence"` — жадная упаковка предложений (default);
  - `"fixed"` — скользящее окно по токенам.

## RetrieverConfig — поиск

```26:42:shkerebert/config.py
@dataclass(frozen=True)
class RetrieverConfig:
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 5
    min_score: float = 0.15       # retrieval-level абстенция
    normalize: bool = True        # L2-норм → cosine = inner product

    use_reranker: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_pool: int = 20

    use_bm25: bool = False
    bm25_pool: int = 50
    rrf_k: int = 60               # константа RRF
```

| Поле | Назначение |
|------|------------|
| `embed_model` | HF-id sentence-transformer |
| `top_k` | Сколько фрагментов отдаётся ридеру |
| `min_score` | Порог косинуса; ниже → «нет ответа» на retrieval-уровне |
| `normalize` | L2-нормировка эмбеддингов перед FAISS |
| `use_reranker` | Cross-encoder переоценка top-pool |
| `use_bm25` | RRF-слияние с BM25 |
| `rrf_k` | Константа Reciprocal Rank Fusion (типично 60) |

## ReaderConfig — экстрактивный BERT

```45:51:shkerebert/config.py
@dataclass(frozen=True)
class ReaderConfig:
    model: str = "deepset/tinyroberta-squad2"
    max_seq_len: int = 384
    max_answer_len: int = 30
    no_answer_threshold: float = -6.3   # τ: gap > τ => «нет ответа»
```

- **max_seq_len** — `[CLS] question [SEP] context [SEP]` усечение; `truncation="only_second"`
  режет контекст, не вопрос.
- **max_answer_len** — максимальная длина span в токенах (перебор start/end).
- **no_answer_threshold (τ)** — эмпирически подобран на calibration-сплите; gap =
  `null_score - best_span_score`.

## GeneratorConfig — Qwen GGUF

```54:62:shkerebert/config.py
@dataclass(frozen=True)
class GeneratorConfig:
    model_path: str = "models/qwen2.5-3b-instruct-q4_k_m.gguf"
    n_ctx: int = 4096
    n_threads: int = 0          # 0 = все доступные
    temperature: float = 0.0
    max_tokens: int = 256
    no_answer_marker: str = "НЕТ ОТВЕТА"
```

`temperature=0.0` — детерминированная генерация для воспроизводимости.

## Фабрики конфига

```83:96:shkerebert/config.py
def default_config() -> Config:
    """Базовая конфигурация (EN, CPU, быстрые модели)."""
    return Config()

def multilingual_config() -> Config:
    """Конфигурация с многоязычным эмбеддером (EN+RU поиск)."""
    return replace(
        default_config(),
        retriever=replace(
            default_config().retriever,
            embed_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ),
    )
```

## with_overrides — точечные изменения

```99:120:shkerebert/config.py
def with_overrides(cfg: Config, **kwargs) -> Config:
    """Переопределить вложенные поля через двойное подчёркивание:
       with_overrides(cfg, chunk__size=128, retriever__top_k=10)
    """
```

Пример:

```python
from shkerebert.config import default_config, with_overrides

cfg = with_overrides(
    default_config(),
    chunk__size=128,
    retriever__top_k=10,
    retriever__use_bm25=True,
    reader__no_answer_threshold=-5.0,
)
```

## Кэш-директория

```75:80:shkerebert/config.py
    def cache_dir(self, subdir: str) -> Path:
        root = Path(os.getenv("SHKEREBERT_DATA", Path.home() / ".cache" / "shkerebert"))
        path = root / subdir
        path.mkdir(parents=True, exist_ok=True)
        return path
```

Эмбеддинги сохраняются в `{SHKEREBERT_DATA}/embeddings/{signature}.npy`.

## Baseline vs stretch

| Параметр | Baseline | Stretch (опционально) |
|----------|----------|----------------------|
| embed_model | all-MiniLM-L6-v2 | paraphrase-multilingual-MiniLM-L12-v2 |
| reader | tinyroberta-squad2 | roberta-base-squad2 |
| use_bm25 | False | True |
| use_reranker | False | True |
| reader_mode | extractive | generative / hybrid |

Baseline выбран по ablation'ам: chunk=256, k=5, extractive — баланс F1/латентность/память.
