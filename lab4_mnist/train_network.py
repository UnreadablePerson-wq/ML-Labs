# Обучение нейронной сети на MNIST
# Задача: определить, является ли цифра на изображении восьмеркой

import numpy as np
import random
import matplotlib.pyplot as plt
import os
from mnist_loader import load_mnist_paired, print_digit
from neural_network import NeuralNetwork

# Конфигурация
HIDDEN_SIZE = 64          # нейронов в скрытом слое
LEARNING_RATE = 0.01      # скорость обучения
EPOCHS = 15               # количество эпох
TRAIN_SAMPLES = 5000      # сколько примеров брать для обучения
TEST_SAMPLES = 1000       # сколько для тестирования
TARGET_DIGIT = 8          # какую цифру распознаем (младший разряд 18)

# Получаем путь к папке, где находится этот скрипт
script_dir = os.path.dirname(os.path.abspath(__file__))

# Пути к файлам
TRAIN_IMAGES = os.path.join(script_dir, 'Mnist', 'train-images-idx3-ubyte.gz')
TRAIN_LABELS = os.path.join(script_dir, 'Mnist', 'train-labels-idx1-ubyte.gz')
TEST_IMAGES = os.path.join(script_dir, 'Mnist', 't10k-images-idx3-ubyte.gz')
TEST_LABELS = os.path.join(script_dir, 'Mnist', 't10k-labels-idx1-ubyte.gz')

def print_sample_digits(images, labels, n=5):
    """
    Выводит несколько примеров цифры 8 для отчета
    """
    print(f"\n{n} примеров цифры {TARGET_DIGIT} из обучающей выборки:")
    print("-" * 50)
    
    # Находим все восьмерки
    eight_indices = np.where(labels == 1)[0]
    
    for i, idx in enumerate(eight_indices[:n]):
        print(f"\nПример {i+1}:")
        print_digit(images[idx])
        print()

def plot_training_history(losses):
    """
    Строит график изменения ошибки
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses) + 1), losses, 'b-', linewidth=2, marker='o')
    plt.title('Изменение ошибки в процессе обучения')
    plt.xlabel('Эпоха')
    plt.ylabel('Средняя ошибка')
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, len(losses) + 1))
    
    # Добавляем значения на график
    for i, loss in enumerate(losses):
        plt.annotate(f'{loss:.4f}', (i+1, loss), 
                    textcoords="offset points", xytext=(0,10), 
                    ha='center', fontsize=8)
    
    # Создаем папку images если её нет
    images_dir = os.path.join(script_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'training_curve.png'), dpi=150, bbox_inches='tight')
    print(f"\nГрафик сохранен в {os.path.join(images_dir, 'training_curve.png')}")
    plt.show(block=True)

def analyze_errors(network, test_images, test_labels, n_examples=5):
    """
    Анализирует примеры, на которых сеть ошибается
    """
    print("\nАнализ ошибок:")
    print("-" * 50)
    
    false_positives = []   # сказал 8, а на самом деле нет
    false_negatives = []   # сказал не 8, а на самом деле 8
    
    for i in range(len(test_images)):
        X = test_images[i]
        y_true = test_labels[i]
        y_pred = network.predict(X)[0, 0]
        
        if y_true == 1 and y_pred == 0:
            false_negatives.append((i, X, y_true))
        elif y_true == 0 and y_pred == 1:
            false_positives.append((i, X, y_true))
    
    print(f"Ложноположительные (сказал 8, а это не 8): {len(false_positives)}")
    print(f"Ложноотрицательные (не сказал 8, а это 8): {len(false_negatives)}")
    
    # Показываем несколько примеров ошибок
    if false_negatives:
        print(f"\nПримеры пропущенных восьмерок:")
        for i, (idx, img, _) in enumerate(false_negatives[:n_examples]):
            print(f"\nОшибка {i+1} (индекс {idx}):")
            print_digit(img)
    
    if false_positives:
        print(f"\nПримеры ложных срабатываний:")
        for i, (idx, img, _) in enumerate(false_positives[:n_examples]):
            print(f"\nОшибка {i+1} (индекс {idx}):")
            print_digit(img)

def main():
    print("=" * 60)
    print("ОБУЧЕНИЕ НЕЙРОННОЙ СЕТИ ДЛЯ РАСПОЗНАВАНИЯ ЦИФРЫ 8")
    print("=" * 60)
    print(f"Параметры:")
    print(f"  Скрытый слой: {HIDDEN_SIZE} нейронов")
    print(f"  Скорость обучения: {LEARNING_RATE}")
    print(f"  Эпох: {EPOCHS}")
    print(f"  Обучающих примеров: {TRAIN_SAMPLES}")
    print(f"  Тестовых примеров: {TEST_SAMPLES}")
    print(f"  Целевая цифра: {TARGET_DIGIT}")
    
    # Загружаем данные
    print("\nЗагрузка обучающих данных...")
    train_images, train_labels = load_mnist_paired(
        TRAIN_IMAGES, TRAIN_LABELS, TARGET_DIGIT, TRAIN_SAMPLES
    )
    
    print("Загрузка тестовых данных...")
    test_images, test_labels = load_mnist_paired(
        TEST_IMAGES, TEST_LABELS, TARGET_DIGIT, TEST_SAMPLES
    )
    
    print(f"Обучающая выборка: {train_images.shape[0]} примеров")
    print(f"  из них восьмерок: {np.sum(train_labels)}")
    print(f"Тестовая выборка: {test_images.shape[0]} примеров")
    print(f"  из них восьмерок: {np.sum(test_labels)}")
    
    # Показываем примеры
    print_sample_digits(train_images, train_labels)
    
    # Создаем сеть
    print("\nИнициализация сети...")
    network = NeuralNetwork(
        input_size=784,
        hidden_size=HIDDEN_SIZE,
        learning_rate=LEARNING_RATE
    )
    
    # Обучение
    print("\nНачинаем обучение...")
    print("-" * 50)
    
    indices = list(range(TRAIN_SAMPLES))
    loss_history = []
    
    for epoch in range(EPOCHS):
        total_loss = 0
        random.shuffle(indices)
        
        for i in indices:
            X = train_images[i]
            y = train_labels[i]
            
            loss = network.train_step(X, y)
            total_loss += loss
        
        avg_loss = total_loss / TRAIN_SAMPLES
        loss_history.append(avg_loss)
        
        print(f"Эпоха {epoch+1:2d}/{EPOCHS}, Ошибка: {avg_loss:.6f}")
    
    print("-" * 50)
    print("Обучение завершено!")
    
    # Тестирование
    print("\nТестирование сети...")
    correct = 0
    for i in range(TEST_SAMPLES):
        X = test_images[i]
        y_true = test_labels[i]
        y_pred = network.predict(X)[0, 0]
        
        if y_pred == y_true:
            correct += 1
    
    accuracy = correct / TEST_SAMPLES * 100
    print(f"\nТочность на тестовой выборке: {accuracy:.2f}%")
    print(f"Цифра {TARGET_DIGIT} распознана верно в {correct} из {TEST_SAMPLES} случаев.")
    
    # Анализ ошибок
    analyze_errors(network, test_images, test_labels)
    
    # График обучения
    print("\nЗакрой график, чтобы продолжить...")
    plot_training_history(loss_history)
    
    # Сохраняем веса
    weights_path = os.path.join(script_dir, 'trained_weights.npz')
    network.save_weights(weights_path)
    print(f"\nВеса сохранены в {weights_path}")
    
    return network, loss_history, accuracy

if __name__ == "__main__":
    network, losses, acc = main()