# Локальные модели (GGUF)

Сюда кладутся веса генеративного ридера (Qwen 2.5 Instruct, формат GGUF). Файлы `*.gguf`
**не** попадают в git (см. `.gitignore`) — они гигабайтные.

Пайплайн по умолчанию ищет здесь файл `qwen2.5-3b-instruct-q4_k_m.gguf`
(см. `shkerebert/config.py` → `GeneratorConfig.model_path`). Путь можно переопределить:
```bash
export SHKEREBERT_LLM=/полный/путь/к/model.gguf
```

## Что положить
- `qwen2.5-3b-instruct-q4_k_m.gguf` (~2 ГБ) — основной генератор (обязательно для
  режимов `generative` / `hybrid`).
- `qwen2.5-7b-instruct-q4_k_m.gguf` (~4.4 ГБ) — опционально, более сильный (медленнее на CPU).

## Скопировать из соседних проектов (на этой машине)
```bash
cp /home/ijstt/home-ai-agent/models/qwen2.5-3b-instruct-q4_k_m.gguf ~/ShkereBERT/models/
# по желанию — 7B:
cp /home/ijstt/home-ai-agent/models/qwen2.5-7b-instruct-q4_k_m.gguf ~/ShkereBERT/models/
```

## Проверка, что модель подхватилась
```bash
PYTHONPATH=. .venv/bin/python -c "from shkerebert.config import default_config; import os; p=default_config().generator.model_path; print(p, 'EXISTS' if os.path.exists(p) else 'НЕ НАЙДЕН')"
```
