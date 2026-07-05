# Методология End-to-End оценки (E2E)

## Цель
Честно измерить качество полного пайплайна (retrieval + reader) по официальной метрике
SQuAD v2 (EM, F1, HasAns_F1, NoAns_F1) **без утечки данных**.

## Протокол: Calibration / Test Split

```
Все вопросы (seed=42)
        │
        ▼
┌───────┴───────┐
│  Shuffle      │
└───────┬───────┘
        │
        ▼
┌───────────────────────┐
│  Calibration (50%)    │  ← подбор τ (порог абстенции)
│  - перебор τ по сетке │
│  - максимизация F1    │
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│  Test (50%)           │  ← отчёт при фиксированном τ*
│  - score_at_tau(τ*)   │
│  - официальные метрики│
└───────────────────────┘
```

**Почему это важно**: в v1 τ калибровался и отчитывался на одной выборке → F1 78.3.
После разделения → честный F1 75.8 (утечка ~2.5 п.). Мы это нашли и исправили сами.

## Реализация (eval/eval_e2e.py)

### 1. Сбор «сырых» предсказаний (один прогон модели)
```python
def collect_raw_predictions(cfg, questions, retriever, reader):
    # Батч-кодируем вопросы, ищем top-k, читаем ридером
    # Сохраняем для каждого вопроса:
    #   - best_span_text
    #   - gap = null_score - best_span_score
    #   - top1_dense_score
    #   - gold answers, is_impossible
    return raw_predictions, latency_ms_per_q
```

### 2. Свип τ на Calibration
```python
def calibrate(cfg, raw, taus=None):
    # taus = np.linspace(percentile(gap, 2), percentile(gap, 98), 31)
    # Для каждого τ: применяем абстенцию (gap > τ или top1 < min_score)
    # Считаем squad_v2 метрики через HF evaluate
    # Возвращаем DataFrame кривой + best_row (max F1)
```

### 3. Оценка на Test при фиксированном τ*
```python
def score_fixed_tau(cfg, raw, tau):
    # Применяем τ* к test-сырым данным, считаем метрики
```

### 4. Oracle Baseline (потолок ридера)
```python
def collect_oracle_predictions(cfg, questions, reader):
    # Ридер читает ЗОЛОТОЙ контекст (context из SQuAD), без retrieval
    # Показывает: какой F1 был бы при идеальном поиске
```

## Метрики в отчёте

| Метрика | Где считается | Что показывает |
|---|---|---|
| **EM / F1** | Test при τ* | Общее качество |
| **HasAns_F1** | Test при τ* | Качество на вопросах с ответом |
| **NoAns_F1** | Test при τ* | Качество отказа (критично для банка) |
| **Oracle F1** | Oracle test | Потолок ридера |
| **Retrieval Loss** | Oracle F1 − Retrieval F1 | Цена этапа поиска |
| **τ*** | Calibration | Оптимальный порог |
| **Latency** | Wall-time | Скорость на CPU |

## График калибровки (`e2e_threshold_curve.png`)

- X: τ (gap threshold)
- Y: F1 (overall, HasAns, NoAns)
- Красная пунктирная линия: τ* (argmax overall F1 на calibration)
- Позволяет видеть трейд-офф: строгий τ → высокий NoAns_F1, низкий HasAns_F1

## Воспроизводимость

```bash
# Полный прогон
./run.sh eval