"""Единый загрузчик всех моделей ShkereBERT (устойчив к обрывам сети).

Запуск:
    PYTHONPATH=. .venv/bin/python scripts/download_models.py            # обязательные + RU-эмбеддер
    PYTHONPATH=. .venv/bin/python scripts/download_models.py --extras   # + reranker и roberta-base

Не зависит от консольной команды huggingface-cli (её может не быть в venv). Использует
huggingface_hub.snapshot_download и datasets. Каждую модель качает с ретраями; при обрыве
докачивает с места. GGUF-модель Qwen не качает — она копируется в models/ вручную (проверяет
наличие). Тяжёлые onnx/openvino-варианты пропускаются.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (repo_id, что это, обязательна ли)
CORE = [
    ("sentence-transformers/all-MiniLM-L6-v2", "эмбеддер EN (baseline)", True),
    ("deepset/tinyroberta-squad2", "экстрактивный ридер (EM/F1)", True),
    ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "эмбеддер EN+RU (поиск)", True),
]
EXTRAS = [
    ("deepset/roberta-base-squad2", "более сильный ридер (сравнение)", False),
    ("cross-encoder/ms-marco-MiniLM-L-6-v2", "reranker (Pool B)", False),
]
IGNORE = ["onnx/*", "openvino/*", "*.onnx", "tf_model.h5", "openvino_model.*"]
QWEN = ROOT / "models" / "qwen2.5-3b-instruct-q4_k_m.gguf"


def fetch(repo: str, retries: int = 8) -> bool:
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

    for attempt in range(retries):
        try:
            snapshot_download(repo, ignore_patterns=IGNORE, etag_timeout=60)
            return True
        except (GatedRepoError, RepositoryNotFoundError) as e:
            print(f"    ! {type(e).__name__}: {e}")
            return False
        except Exception as e:
            wait = min(5 + attempt * 3, 20)
            print(f"    …обрыв ({type(e).__name__}); повтор через {wait}s "
                  f"[{attempt + 1}/{retries}]")
            time.sleep(wait)
    return False


def fetch_dataset(retries: int = 8) -> bool:
    for attempt in range(retries):
        try:
            from datasets import load_dataset
            load_dataset("rajpurkar/squad_v2")
            return True
        except Exception as e:
            wait = min(5 + attempt * 3, 20)
            print(f"    …обрыв ({type(e).__name__}); повтор через {wait}s "
                  f"[{attempt + 1}/{retries}]")
            time.sleep(wait)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extras", action="store_true", help="скачать и опциональные модели")
    args = ap.parse_args()

    os.environ.setdefault("USE_TF", "0")
    items = CORE + (EXTRAS if args.extras else [])
    ok, fail = [], []

    for repo, desc, _ in items:
        print(f"→ {repo}  ({desc})")
        (ok if fetch(repo) else fail).append(repo)

    print("→ датасет rajpurkar/squad_v2  (замеры качества)")
    (ok if fetch_dataset() else fail).append("rajpurkar/squad_v2")

    print("\n=== ИТОГ ===")
    for r in ok:
        print(f"  ✓ {r}")
    for r in fail:
        print(f"  ✗ {r}  — повтори запуск скрипта (докачает с места обрыва)")

    print("\n=== GGUF (Qwen, копируется вручную) ===")
    if QWEN.exists():
        print(f"  ✓ {QWEN}  ({QWEN.stat().st_size / 1e9:.1f} GB)")
    else:
        print(f"  ✗ нет {QWEN}")
        print("     cp /home/ijstt/home-ai-agent/models/qwen2.5-3b-instruct-q4_k_m.gguf models/")

    print("\nГотово." if not fail else "\nЧасть моделей не докачалась — просто запусти скрипт ещё раз.")


if __name__ == "__main__":
    main()
