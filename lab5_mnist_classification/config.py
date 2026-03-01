# config.py
import os

# Пути к данным - исправлено на Mnist с заглавной буквы
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'Mnist')  # Изменено с 'data' на 'Mnist'

# Файлы датасета
TRAIN_IMAGES = os.path.join(DATA_PATH, 'train-images-idx3-ubyte.gz')
TRAIN_LABELS = os.path.join(DATA_PATH, 'train-labels-idx1-ubyte.gz')
TEST_IMAGES = os.path.join(DATA_PATH, 't10k-images-idx3-ubyte.gz')
TEST_LABELS = os.path.join(DATA_PATH, 't10k-labels-idx1-ubyte.gz')

# Базовая модель
BASE_LR = 0.05
BASE_HIDDEN = 25
BASE_EPOCHS = 20

# Улучшенная модель (подобранные параметры)
BEST_LR = 0.01
BEST_HIDDEN = 64
BEST_EPOCHS = 20

# Для автоматического подбора можно добавить сетку параметров
GRID_SEARCH = {
    'learning_rates': [0.001, 0.01, 0.05, 0.1],
    'hidden_sizes': [32, 64, 128, 256],
    'epochs': 15
}

# Результаты
RESULTS_DIR = os.path.join(BASE_DIR, 'results')