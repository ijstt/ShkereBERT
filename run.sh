#!/usr/bin/env bash
# Простой запуск ShkereBERT. Сам ставит venv, PYTHONPATH и нужные переменные.
#
#   ./run.sh web                 — веб-интерфейс (http://localhost:8501)
#   ./run.sh chat [ФАЙЛ]         — CLI-чат, extractive (по умолчанию demo-документ EN)
#   ./run.sh gen  [ФАЙЛ]         — CLI-чат, generative (Qwen)
#   ./run.sh ru   [ФАЙЛ]         — RU: многоязычный поиск + generative (по умолчанию банк-демо)
#   ./run.sh ask  "ВОПРОС" [ФАЙЛ] [РЕЖИМ]
#   ./run.sh test                — прогнать тесты
#   ./run.sh eval                — оценка (retrieval + e2e)
#   ./run.sh eval-extra          — долгие исследования (longdoc/multiseed/варианты/генерация)
#   ./run.sh check               — проверить окружение (venv, модели)

set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
export PYTHONPATH="$PWD"
RU_EMBED="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

if [[ ! -x "$PY" ]]; then
  echo "Нет venv. Создай: python3 -m venv --system-site-packages .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi

cmd="${1:-help}"; shift || true

case "$cmd" in
  web)
    exec .venv/bin/streamlit run app/streamlit_app.py
    ;;
  chat)
    file="${1:-demo/machine_learning.txt}"
    exec $PY -m shkerebert.cli chat --file "$file" --mode extractive
    ;;
  gen)
    file="${1:-demo/machine_learning.txt}"
    exec $PY -m shkerebert.cli chat --file "$file" --mode generative
    ;;
  ru)
    file="${1:-demo/bank_products_ru.txt}"
    export SHKEREBERT_EMBED="$RU_EMBED"
    exec $PY -m shkerebert.cli chat --file "$file" --mode generative
    ;;
  ask)
    q="${1:?нужен вопрос в кавычках}"
    file="${2:-demo/machine_learning.txt}"
    mode="${3:-extractive}"
    exec $PY -m shkerebert.cli ask "$q" --file "$file" --mode "$mode"
    ;;
  test)
    exec $PY -m pytest -q
    ;;
  eval)
    $PY -m eval.eval_retrieval --n 800
    $PY -m eval.eval_e2e --n 1500
    ;;
  eval-extra)   # долгие исследования: длинные документы, multi-seed CI, варианты поиска, генерация
    $PY -u -m eval.eval_longdoc --n 600
    $PY -u -m eval.eval_multiseed --n 800
    $PY -u -m eval.eval_retrieval_variants --n 800
    $PY -u -m eval.eval_generative --n 150
    ;;
  check)
    echo "venv:      $($PY --version)"
    echo "PYTHONPATH: $PYTHONPATH"
    $PY - <<'PY'
import os
from shkerebert.config import default_config
c = default_config()
llm = c.generator.model_path
print("Qwen GGUF:", llm, "->", "НАЙДЕН" if os.path.exists(llm) else "НЕ НАЙДЕН (скопируй в models/)")
for name in ["sentence-transformers/all-MiniLM-L6-v2",
             "deepset/tinyroberta-squad2",
             "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"]:
    p = os.path.expanduser("~/.cache/huggingface/hub/models--" + name.replace("/", "--"))
    print(f"{'есть ' if os.path.isdir(p) else 'нет  '} {name}")
PY
    ;;
  help|*)
    sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
    ;;
esac
