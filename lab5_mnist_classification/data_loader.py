# data_loader.py
import numpy as np
import idx2numpy
import gzip
import os

def load_mnist():
    """
    Загружает обучающую и тестовую выборки MNIST из .gz файлов
    """
    # Пути к файлам (как в исходной программе, но адаптированные под текущую структуру)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    mnist_dir = os.path.join(base_dir, 'Mnist')
    
    TRAIN_IMAGE_FILENAME = os.path.join(mnist_dir, 'train-images-idx3-ubyte.gz')
    TRAIN_LABEL_FILENAME = os.path.join(mnist_dir, 'train-labels-idx1-ubyte.gz')
    TEST_IMAGE_FILENAME = os.path.join(mnist_dir, 't10k-images-idx3-ubyte.gz')
    TEST_LABEL_FILENAME = os.path.join(mnist_dir, 't10k-labels-idx1-ubyte.gz')
    
    # Проверка наличия файлов
    for path in [TRAIN_IMAGE_FILENAME, TRAIN_LABEL_FILENAME, TEST_IMAGE_FILENAME, TEST_LABEL_FILENAME]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Файл не найден: {path}")
    
    # Распаковка и чтение .gz файлов
    print("Распаковка и загрузка файлов...")
    
    with gzip.open(TRAIN_IMAGE_FILENAME, 'rb') as f:
        train_images = idx2numpy.convert_from_file(f)
    
    with gzip.open(TRAIN_LABEL_FILENAME, 'rb') as f:
        train_labels = idx2numpy.convert_from_file(f)
    
    with gzip.open(TEST_IMAGE_FILENAME, 'rb') as f:
        test_images = idx2numpy.convert_from_file(f)
    
    with gzip.open(TEST_LABEL_FILENAME, 'rb') as f:
        test_labels = idx2numpy.convert_from_file(f)
    
    print(f"Загружено: {len(train_images)} обучающих, {len(test_images)} тестовых примеров")
    
    return train_images, train_labels, test_images, test_labels

def prepare_data(train_images, train_labels, test_images, test_labels):
    """
    Подготовка данных: нормализация, one-hot кодирование.
    Возвращает подготовленные массивы и статистики для нормализации.
    """
    # Преобразование в векторы
    x_train = train_images.reshape(60000, 784).astype(np.float32)
    x_test = test_images.reshape(10000, 784).astype(np.float32)
    
    # Статистики для нормализации
    mean = np.mean(x_train)
    std = np.std(x_train)
    
    print(f"Статистики нормализации: mean={mean:.4f}, std={std:.4f}")
    
    # Нормализация
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    
    # One-hot кодирование меток
    y_train = np.zeros((60000, 10))
    y_test = np.zeros((10000, 10))
    
    for i, label in enumerate(train_labels):
        y_train[i, label] = 1
    for i, label in enumerate(test_labels):
        y_test[i, label] = 1
    
    return x_train, y_train, x_test, y_test, mean, std

def add_bias(x):
    """Добавляет единичный bias к входному вектору"""
    return np.concatenate(([1.0], x))