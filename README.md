# Image Convolution CLI Utility

Утилита командной строки для применения фильтров свёртки к одиночным изображениям с поддержкой многопоточности и настройки параметров распараллеливания OpenMP.

---

## Синтаксис запуска

```bash
./image_conv_cli [--kernel <name>] [--partition <name>] [--threads <n>] [--schedule <name>] <input.png> <output.png>

```

---

## Параметры и флаги

| Параметр | Описание |
| --- | --- |
| `--kernel <name>` | Название ядра свёртки (например: `blur_3x3`, `blur_5x5`, `motion_blur_9x9`, `identity_3x3`). |
| `--partition <name>` | Стратегия параллельного разбиения изображения (`row`, `column`, `pixel`, `tile`). |
| `--threads <n>` | Количество рабочих потоков OpenMP (число $> 0$). |
| `--schedule <name>` | Тип планировщика распределения итераций OpenMP (`static`, `dynamic`, `guided`). |
| `<input.png>` | Путь к исходному файлу изображения. |
| `<output.png>` | Путь для сохранения обработанного файла. |

---

## Примеры использования

### 1. Запуск с параметрами по умолчанию

Обработка файла с дефолтным ядром и автоматическим выбором параметров OpenMP:

```bash
./image_conv_cli input.png output.png

```

### 2. Выбор ядра и стратегии разбиения

Применить ядро `blur_5x5` с разбиением по строкам (`row`):

```bash
./image_conv_cli --kernel blur_5x5 --partition row input.png output.png

```

### 3. Полный контроль над потоками и планировщиком

Применить ядро `motion_blur_9x9` на 8 потоках с тайловым разбиением (`tile`) и динамическим планировщиком (`dynamic`):

```bash
./image_conv_cli --kernel motion_blur_9x9 --partition tile --threads 8 --schedule dynamic input.png output.png

```

---

## Получение справки

Запустите программу без аргументов или с неверным параметром, чтобы вывести список всех поддерживаемых в данной сборке ядер, разбиений и планировщиков:

```bash
./image_conv_cli

```