# 13. Интерфейсы

## CLI — Typer (`shkerebert/cli.py`)

Typer — обёртка над argparse/click с type hints и автоматической `--help`.

```17:17:shkerebert/cli.py
app = typer.Typer(add_completion=False, help="ShkereBERT — чат-бот по документу (RAG).")
```

### _build — сборка пайплайна

```20:29:shkerebert/cli.py
def _build(file, squad_n):
    cfg = default_config()
    if file:
        docs = [load_document(file)]
    else:
        docs = load_squad_documents(split="validation", max_contexts=squad_n)
    return RAGPipeline.build(docs, cfg)
```

Всегда `default_config()` — без multilingual/generator overrides из env.

### chat — интерактивный цикл

```52:68:shkerebert/cli.py
@app.command()
def chat(file=..., squad_n=200, mode="extractive"):
    pipe = _build(file, squad_n)
    while True:
        q = typer.prompt("Вопрос")
        if q.strip().lower() in {"exit", "quit", ""}:
            break
        _print_answer(pipe.answer(q, mode=mode))
```

### ask — одиночный вопрос

Для скриптов и автоматизации:

```bash
python -m shkerebert.cli ask "What is X?" --file doc.txt --mode hybrid
```

### _print_answer — формат вывода

- Зелёный текст — answerable ответ;
- Жёлтый — «нет ответа»;
- Строка `[mode] (confidence=...; reason)`;
- До 3 источников со snippet 160 символов, `★` на answer source.

---

## Streamlit UI (`app/streamlit_app.py`)

### Bootstrap

```18:19:app/streamlit_app.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

Позволяет запускать без `pip install -e .` — достаточно `PYTHONPATH=.` или запуск
из корня через `run.sh`.

### Кэширование пайплайна

```28:37:app/streamlit_app.py
@st.cache_resource(show_spinner=True)
def build_pipeline_squad(n: int):
    docs = load_squad_documents(split="validation", max_contexts=n)
    return RAGPipeline.build(docs, default_config(), show_progress=False)

@st.cache_resource(show_spinner=True)
def build_pipeline_file(path: str):
    docs = [load_document(path)]
    return RAGPipeline.build(docs, default_config(), show_progress=False)
```

`@st.cache_resource` — пайплайн пересобирается только при смене ключа (squad_n или path).

### Sidebar

1. **Источник:** SQuAD slider (50–1000) или file_uploader (PDF/TXT).
2. Загруженный файл сохраняется в `data/upload_{name}`.
3. **Режим ридера:** radio extractive / generative / hybrid.

### Основная область

```80:95:app/streamlit_app.py
    ans = pipe.answer(question, mode=reader_mode)
    if ans.is_answerable:
        st.success(f"**Ответ:** {ans.answer}")
    else:
        st.warning("**В документе нет ответа на этот вопрос.**")
    ...
    for s in ans.sources[:5]:
        with st.expander(title, expanded=s.is_answer_source):
            st.write(s.text)
```

---

## run.sh — convenience wrapper

Bash-скрипт в корне репозитория:

| Команда | Действие |
|---------|----------|
| `web` | `streamlit run app/streamlit_app.py` |
| `chat` | CLI extractive, default `demo/machine_learning.txt` |
| `gen` | CLI generative |
| `ru` | generative + `demo/bank_products_ru.txt` |
| `ask` | один вопрос |
| `test` | `pytest -q` |
| `eval` | retrieval + e2e |
| `eval-extra` | longdoc, multiseed, variants, generative |
| `check` | версия Python, наличие моделей |

Устанавливает `PYTHONPATH=$PWD` и использует `.venv/bin/python`.

---

## Сравнение интерфейсов

| | CLI | Streamlit |
|---|-----|-----------|
| Сборка индекса | каждый запуск | cache_resource |
| Режимы ридера | `--mode` | sidebar radio |
| Источники | 3 snippet | 5 expander с полным текстом |
| SQuAD / файл | `--squad-n` / `--file` | sidebar |
| Generative | да | да (+ предупреждение о задержке) |

Оба вызывают один и тот же `RAGPipeline.answer()` — продуктовый путь идентичен eval
(кроме `eval_generative.py`, который тоже идёт через pipeline).
