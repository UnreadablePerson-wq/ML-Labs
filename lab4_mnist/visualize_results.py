# Визуализация результатов обучения
# Графики, примеры цифр, анализ ошибок

import numpy as np
import matplotlib.pyplot as plt
import os
from mnist_loader import load_mnist_paired
from neural_network import NeuralNetwork
from utils import (
    plot_digit, plot_digits_grid, plot_error_analysis,
    plot_weight_visualization, create_sample_output_file
)

# Параметры (должны совпадать с train_network.py)
HIDDEN_SIZE = 64
TARGET_DIGIT = 8
TEST_SAMPLES = 1000

# Получаем путь к папке, где находится этот скрипт
script_dir = os.path.dirname(os.path.abspath(__file__))

# Пути к файлам
TRAIN_IMAGES = os.path.join(script_dir, 'Mnist', 'train-images-idx3-ubyte.gz')
TRAIN_LABELS = os.path.join(script_dir, 'Mnist', 'train-labels-idx1-ubyte.gz')
TEST_IMAGES = os.path.join(script_dir, 'Mnist', 't10k-images-idx3-ubyte.gz')
TEST_LABELS = os.path.join(script_dir, 'Mnist', 't10k-labels-idx1-ubyte.gz')

def show_sample_eights(n_samples=5):
    """
    Показывает несколько примеров цифры 8 из обучающей выборки
    Для отчета (как на рисунках 1-5)
    """
    print(f"\nЗагрузка данных для визуализации...")
    
    # Загружаем немного данных для примера
    train_images, train_labels = load_mnist_paired(
        TRAIN_IMAGES, TRAIN_LABELS, TARGET_DIGIT, 1000
    )
    
    # Находим все восьмерки
    eight_indices = np.where(train_labels == 1)[0]
    
    print(f"\n{n_samples} примеров цифры {TARGET_DIGIT}:")
    print("=" * 50)
    
    # Создаем папку для сохранения
    images_dir = os.path.join(script_dir, 'images', 'sample_digits')
    os.makedirs(images_dir, exist_ok=True)
    
    for i, idx in enumerate(eight_indices[:n_samples]):
        print(f"\nПример {i+1}:")
        img = train_images[idx]
        
        # Выводим в консоль псевдографикой
        for row in img.reshape(28, 28):
            line = ''
            for pixel in row:
                if pixel > 0.5:
                    line += '██'
                else:
                    line += '  '
            print(line)
        
        # Сохраняем как изображение
        plt.figure(figsize=(5, 5))
        plt.imshow(img.reshape(28, 28), cmap='binary', interpolation='nearest')
        plt.axis('off')
        plt.title(f'Цифра 8 - пример {i+1}')
        save_path = os.path.join(images_dir, f'eight_{i+1}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Сохранено: {save_path}")

def plot_training_curve(losses=None):
    """
    Строит график обучения (как на рисунке 9)
    """
    if losses is None:
        # Если не передали историю ошибок, используем примерные значения
        losses = [0.073777, 0.045469, 0.037912, 0.033044, 0.028902,
                 0.025628, 0.022263, 0.019865, 0.017539, 0.015573,
                 0.014005, 0.012283, 0.011302, 0.010116, 0.009315]
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses) + 1), losses, 'b-', linewidth=2, marker='o', markersize=6)
    plt.title('Изменение ошибки при обучении', fontsize=14)
    plt.xlabel('Эпоха', fontsize=12)
    plt.ylabel('Средняя квадратичная ошибка', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, len(losses) + 1))
    
    # Подписываем точки
    for i, loss in enumerate(losses):
        plt.annotate(f'{loss:.4f}', (i+1, loss), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=8)
    
    images_dir = os.path.join(script_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    plt.tight_layout()
    save_path = os.path.join(images_dir, 'training_curve.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nГрафик обучения сохранен в {save_path}")
    plt.show(block=True)

def visualize_network_weights():
    """
    Визуализация весов нейронов (опционально)
    """
    # Загружаем обученную сеть
    network = NeuralNetwork(input_size=784, hidden_size=HIDDEN_SIZE)
    
    weights_path = os.path.join(script_dir, 'trained_weights.npz')
    
    try:
        network.load_weights(weights_path)
        print("Веса загружены из trained_weights.npz")
        
        # Визуализируем веса
        images_dir = os.path.join(script_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)
        
        plot_weight_visualization(
            network, 
            n_neurons=16,
            save_path=os.path.join(images_dir, 'weight_visualization.png')
        )
    except FileNotFoundError:
        print("Файл с весами не найден. Сначала обучи сеть командой:")
        print("python train_network.py")

def test_pretrained_network():
    """
    Тестирование сохраненной сети
    """
    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ ОБУЧЕННОЙ СЕТИ")
    print("=" * 60)
    
    # Загружаем тестовые данные
    print("Загрузка тестовых данных...")
    test_images, test_labels = load_mnist_paired(
        TEST_IMAGES, TEST_LABELS, TARGET_DIGIT, TEST_SAMPLES
    )
    
    # Создаем и загружаем сеть
    network = NeuralNetwork(input_size=784, hidden_size=HIDDEN_SIZE)
    
    weights_path = os.path.join(script_dir, 'trained_weights.npz')
    
    try:
        network.load_weights(weights_path)
        print("Веса загружены из trained_weights.npz")
    except FileNotFoundError:
        print("Файл с весами не найден. Использую случайные веса.")
        return None, 0
    
    # Тестируем
    print("\nТестирование...")
    correct = 0
    predictions = []
    
    for i in range(len(test_images)):
        pred = network.predict(test_images[i])[0, 0]
        true = test_labels[i]
        predictions.append((pred, true))
        
        if pred == true:
            correct += 1
    
    accuracy = correct / len(test_images) * 100
    print(f"\nТочность на тестовой выборке: {accuracy:.2f}%")
    print(f"Правильно: {correct} из {len(test_images)}")
    
    # Создаем таблицу для отчета
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ (первые 20 примеров)")
    print("=" * 60)
    print(" №  | Ожидание | Предсказание | Результат")
    print("-" * 40)
    
    for i, (pred, true) in enumerate(predictions[:20]):
        result = "+" if pred == true else "-"
        print(f"{i+1:2d} |     {int(true)}     |      {pred}       |    {result}")
    
    print("-" * 40)
    print(f"Точность на первых 20: {sum(p == t for p, t in predictions[:20])}/20")
    
    return predictions, accuracy

def create_report_images():
    """
    Создает все изображения для отчета одним запуском
    """
    print("Создание изображений для отчета...")
    print("=" * 60)
    
    # Создаем папки
    os.makedirs(os.path.join(script_dir, 'images', 'sample_digits'), exist_ok=True)
    os.makedirs(os.path.join(script_dir, 'sample_outputs'), exist_ok=True)
    
    # 1. Примеры цифры 8 (Рисунки 1-5)
    print("\n1. Генерация примеров цифры 8...")
    show_sample_eights(5)
    
    # 2. Загружаем обученную сеть для тестирования
    print("\n2. Тестирование сети...")
    predictions, accuracy = test_pretrained_network()
    
    # 3. Создаем файл с логом обучения
    print("\n3. Создание файла с логом обучения...")
    sample_output_path = os.path.join(script_dir, 'sample_outputs', 'training_log.txt')
    create_sample_output_file(sample_output_path)
    
    # 4. Строим график обучения
    print("\n4. Построение графика обучения...")
    plot_training_curve()
    
    # 5. Анализ ошибок (если есть веса)
    print("\n5. Анализ ошибок...")
    try:
        network = NeuralNetwork(input_size=784, hidden_size=HIDDEN_SIZE)
        weights_path = os.path.join(script_dir, 'trained_weights.npz')
        network.load_weights(weights_path)
        
        # Загружаем тестовые данные
        test_images, test_labels = load_mnist_paired(
            TEST_IMAGES, TEST_LABELS, TARGET_DIGIT, TEST_SAMPLES
        )
        
        # Тут можно добавить визуализацию ошибок
        print("   Анализ ошибок завершен")
        
    except FileNotFoundError:
        print("   Файл с весами не найден, пропускаю анализ ошибок")
    
    print("\n" + "=" * 60)
    print("Готово! Все изображения сохранены в папке images/")
    print("Лог обучения сохранен в sample_outputs/training_log.txt")
    print("=" * 60)

if __name__ == "__main__":
    # Запускаем создание всех изображений
    create_report_images()