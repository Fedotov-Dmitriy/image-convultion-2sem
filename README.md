# Image Convolution

Учебный проект по реализации свёртки изображений на Python с использованием NumPy.

Проект позволяет применять к изображению различные ядра свёртки: размытие, повышение резкости, выделение границ. Также реализованы разные способы обработки краёв изображения.

## Возможности

- Свёртка grayscale и RGB-изображений
- Поддержка ядер 3x3 и 5x5
- Режимы обработки краёв:
  - reflect
  - replicate
  - wrap
- CLI-интерфейс для выбора изображения, фильтра и режима обработки краёв
- Тесты с эталонными изображениями
- Бенчмарки производительности Custom / OpenCV / Pillow

## Структура проекта

- `src/` - основной код проекта
  - `main.py` - точка входа
  - `convolution.py` - реализация свёртки
  - `kernels.py` - ядра свёртки
  - `proccesing.py` - обработка краёв изображения
  - `cli.py` - CLI-интерфейс
- `input/` - исходные изображения
- `tests/` - тесты проекта
  - `reference/` - эталонные изображения для тестов
  - `benchmark/` - бенчмарки, результаты и график

## Установка

- Проект использует `uv`
- Установка зависимостей:
  - `uv sync`

## Запуск

- Запуск программы:
  - `uv run python -m src.main`
- После запуска программа предложит выбрать:
  - изображение из папки `input`
  - ядро свёртки
  - режим обработки края
  - цветовой режим

## Тесты

- Запуск основных тестов:
  - `uv run python -m pytest tests/test_conv.py`

## Бенчмарки

- Запуск бенчмарков:
  - `uv run python -m pytest tests/benchmark/test_benchmark.py --benchmark-json=tests/benchmark/results.json`
- Построение графика:
  - `uv run python tests/benchmark/bench_visual.py tests/benchmark/results.json`
- График сохраняется в:
  - `tests/benchmark/benchmark_bar.png`
- Анализ результатов находится в:
  - `benchmark_analysis.md`

## Реализованные ядра

- `box_blur`
- `gaussian_blur`
- `sobel_x`
- `prewitt_y`
- `laplacian`
- `sharpen`
- `sobel_x_5`
- `sobel_y_5`
- `laplacian_5`
- `sharpen_5`
