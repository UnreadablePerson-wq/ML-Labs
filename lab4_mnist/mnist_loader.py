# Загрузка данных MNIST из сжатых файлов .gz
# Формат файлов описан на сайте Yann LeCun

import numpy as np
import gzip
import os

def load_mnist_images(filename, num_samples=None):
    """
    Загружает изображения из файла MNIST в формате idx3-ubyte
    """
    with gzip.open(filename, 'rb') as f:
        # Читаем заголовок
        magic = f.read(4)
        n_images = int.from_bytes(f.read(4), 'big')
        n_rows = int.from_bytes(f.read(4), 'big')
        n_cols = int.from_bytes(f.read(4), 'big')
        
        # Читаем все изображения
        buffer = f.read()
        images = np.frombuffer(buffer, dtype=np.uint8)
        
        # Преобразуем в массив (n_images, 784)
        images = images.reshape(n_images, n_rows * n_cols)
        
        # Нормализуем к диапазону [0, 1]
        images = images.astype(np.float32) / 255.0
        
        # Если нужно, берем только часть данных
        if num_samples and num_samples < n_images:
            indices = np.random.choice(n_images, num_samples, replace=False)
            images = images[indices]
            
    return images

def load_mnist_labels(filename, target_digit, num_samples=None):
    """
    Загружает метки из файла MNIST в формате idx1-ubyte
    Преобразует в бинарную задачу: 1 для target_digit, 0 для остальных
    """
    with gzip.open(filename, 'rb') as f:
        # Читаем заголовок
        magic = f.read(4)
        n_labels = int.from_bytes(f.read(4), 'big')
        
        # Читаем все метки
        buffer = f.read()
        labels = np.frombuffer(buffer, dtype=np.uint8)
        
        # Преобразуем в бинарные метки
        binary_labels = (labels == target_digit).astype(np.float32)
        
        # Если нужно, берем только часть данных
        if num_samples and num_samples < n_labels:
            indices = np.random.choice(n_labels, num_samples, replace=False)
            binary_labels = binary_labels[indices]
            
    return binary_labels

def load_mnist_paired(images_path, labels_path, target_digit, num_samples):
    """
    Загружает изображения и соответствующие им метки
    Возвращает пары (изображение, метка) с одинаковыми индексами
    """
    with gzip.open(labels_path, 'rb') as f:
        # Пропускаем заголовок меток
        f.read(8)
        labels_buffer = f.read()
        all_labels = np.frombuffer(labels_buffer, dtype=np.uint8)
    
    with gzip.open(images_path, 'rb') as f:
        # Пропускаем заголовок изображений
        f.read(16)
        images_buffer = f.read()
        all_images = np.frombuffer(images_buffer, dtype=np.uint8)
        all_images = all_images.reshape(len(all_labels), 784)
    
    # Выбираем случайные индексы
    indices = np.random.choice(len(all_labels), num_samples, replace=False)
    
    images = all_images[indices].astype(np.float32) / 255.0
    labels = (all_labels[indices] == target_digit).astype(np.float32)
    
    return images, labels

def print_digit(image_flat, threshold=0.5):
    """
    Печатает цифру в консоль символами
    Использую для демонстрации в отчете
    """
    image = image_flat.reshape(28, 28)
    for row in image:
        line = ''
        for pixel in row:
            if pixel > threshold:
                line += '██'  # закрашенный пиксель
            else:
                line += '  '  # пустой пиксель
        print(line)

if __name__ == "__main__":
    # Тестирование загрузки
    print("Проверка загрузки MNIST...")
    
    # Получаем путь к папке, где находится этот скрипт
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Пути к файлам (относительно папки скрипта)
    train_images_path = os.path.join(script_dir, 'Mnist', 'train-images-idx3-ubyte.gz')
    train_labels_path = os.path.join(script_dir, 'Mnist', 'train-labels-idx1-ubyte.gz')
    
    print(f"Ищем файлы в: {train_images_path}")
    
    if os.path.exists(train_images_path):
        images, labels = load_mnist_paired(
            train_images_path, 
            train_labels_path, 
            target_digit=8, 
            num_samples=5
        )
        
        print("Загружено 5 примеров цифры 8:")
        for i in range(len(images)):
            print(f"\nПример {i+1}, метка: {int(labels[i])}")
            print_digit(images[i])
    else:
        print(f"Файл не найден!")
        print("Создай папку Mnist и положи туда файлы:")
        print("  - train-images-idx3-ubyte.gz")
        print("  - train-labels-idx1-ubyte.gz")
        print("  - t10k-images-idx3-ubyte.gz")
        print("  - t10k-labels-idx1-ubyte.gz")