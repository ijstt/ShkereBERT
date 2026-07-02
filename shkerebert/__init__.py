"""ShkereBERT — учебный RAG-чат-бот по документу (SQuAD v2).

Экстрактивный BERT-reader поверх dense-retrieval с обработкой случая «ответа нет».
"""

import os

# Работаем только на PyTorch. Отключаем TF/Flax-бэкенды transformers ДО их импорта,
# иначе на машинах с Keras 3 sentence-transformers падает при подтягивании TF-ветки.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

__version__ = "0.1.0"
