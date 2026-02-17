# Обучение базовых логических элементов (AND, OR, NAND)
# Использую NumPy для матричных операций

import numpy as np
import matplotlib.pyplot as plt
import random

# Настройки обучения
LEARNING_RATE = 0.1
random.seed(7)  # чтобы результаты повторялись

# Цвета для линий на графиках (каждая эпоха своим цветом)
colors = ['r-', 'm-', 'y-', 'c-', 'b-', 'g-']
color_idx = 0

def prepare_data(x_train, y_train):
    """
    Разделяем точки по классам для отрисовки
    x_train должен быть размером (n, 3) где первый столбец - единицы
    """
    if x_train.shape[1] != 3:
        return [], []
    
    class_plus = [[], []]  # для y = 1
    class_minus = [[], []] # для y = 0
    
    for i in range(len(x_train)):
        if y_train[i] == 1:
            class_plus[0].append(x_train[i][1])  # x1
            class_plus[1].append(x_train[i][2])  # x2
        else:
            class_minus[0].append(x_train[i][1])
            class_minus[1].append(x_train[i][2])
    
    return class_plus, class_minus

def draw_learning(x_train, y_train, weights, first_draw=False, title=""):
    """
    Рисуем текущее состояние обучения
    weights - [w0, w1, w2]
    """
    global color_idx
    
    if x_train.shape[1] != 3:
        return
    
    # Точки для линии разделения (x1 от -0.5 до 1.5)
    x1_line = [-0.5, 1.5]
    
    # Из уравнения w0 + w1*x1 + w2*x2 = 0 выражаем x2
    if abs(weights[2]) < 1e-5:  # защита от деления на ноль
        x2_line = [0, 0]
    else:
        x2_line = [(-weights[0] - weights[1]*x1_line[0]) / weights[2],
                   (-weights[0] - weights[1]*x1_line[1]) / weights[2]]
    
    if first_draw:
        # Рисуем точки обучающей выборки
        class_plus, class_minus = prepare_data(x_train, y_train)
        plt.plot(class_plus[0], class_plus[1], 'r+', markersize=12, label='Класс 1')
        plt.plot(class_minus[0], class_minus[1], 'b_', markersize=12, label='Класс 0')
        
        # Настройки графика
        plt.axis([-0.5, 1.5, -0.5, 1.5])
        plt.title(title)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.grid(True, alpha=0.3)
    
    # Рисуем линию разделения
    plt.plot(x1_line, x2_line, colors[color_idx], linewidth=2)
    
    # Меняем цвет для следующей линии
    color_idx = (color_idx + 1) % len(colors)
    
    plt.pause(0.3)

def compute_output(weights, inputs):
    """
    Вычисляем выход персептрона
    Использую скалярное произведение (векторно)
    """
    # Скалярное произведение весов и входов
    z = np.dot(weights, inputs)
    
    # Пороговая функция активации
    return 1 if z >= 0 else 0

def train_perceptron(x_train, y_train, title=""):
    """
    Обучаем персептрон на заданных данных
    Возвращает обученные веса
    """
    # Инициализируем веса случайно
    weights = np.random.random(x_train.shape[1])
    
    # Индексы для перемешивания
    indices = list(range(len(x_train)))
    
    print(f"\nОбучение {title}...")
    print("Начальные веса:", [f"{w:.4f}" for w in weights])
    
    # Создаем новый график
    plt.figure(figsize=(8, 6))
    
    # Рисуем начальное состояние
    draw_learning(x_train, y_train, weights, first_draw=True, title=title)
    
    all_correct = False
    epoch = 0
    
    while not all_correct:
        epoch += 1
        all_correct = True
        random.shuffle(indices)
        
        for i in indices:
            x = x_train[i]
            target = y_train[i]
            
            output = compute_output(weights, x)
            
            if output != target:
                # Обновляем веса по дельта-правилу
                for j in range(len(weights)):
                    weights[j] += (target - output) * LEARNING_RATE * x[j]
                all_correct = False
                draw_learning(x_train, y_train, weights, title=title)
        
        if all_correct:
            print(f"Сошлось за {epoch} эпох")
    
    # Показываем график процесса обучения и ждем закрытия
    plt.legend()
    plt.show(block=True)  # ждем закрытия
    
    return weights

def print_weights(weights):
    """Красивый вывод весов"""
    for i, w in enumerate(weights):
        print(f"w{i} = {w:<10.4f}", end="")
    print()

def print_truth_table(x_train, weights, gate_name):
    """
    Проверяем обученный персептрон на всех входах
    """
    print(f"\nТаблица истинности для {gate_name}")
    print("   X1    X2    Y")
    
    for i in range(len(x_train)):
        x1 = int(x_train[i][1])
        x2 = int(x_train[i][2])
        output = compute_output(weights, x_train[i])
        print(f"   {x1}     {x2}     {output}")
    print()

if __name__ == "__main__":
    # Данные для обучения логических элементов
    # Формат: [bias, x1, x2]
    x_data = np.array([
        [1.0, 0, 0],
        [1.0, 0, 1],
        [1.0, 1, 0],
        [1.0, 1, 1]
    ])
    
    # AND
    y_and = np.array([0, 0, 0, 1])
    w_and = train_perceptron(x_data, y_and, "AND")
    
    # Финальный график для AND
    color_idx = 0  # сбрасываем счетчик цветов
    plt.figure(figsize=(8, 6))
    draw_learning(x_data, y_and, w_and, first_draw=True, title="AND - результат")
    plt.legend()
    plt.show(block=True)
    
    print("Веса AND:")
    print_weights(w_and)
    print_truth_table(x_data, w_and, "AND")
    
    # OR
    y_or = np.array([0, 1, 1, 1])
    w_or = train_perceptron(x_data, y_or, "OR")
    
    color_idx = 0
    plt.figure(figsize=(8, 6))
    draw_learning(x_data, y_or, w_or, first_draw=True, title="OR - результат")
    plt.legend()
    plt.show(block=True)
    
    print("Веса OR:")
    print_weights(w_or)
    print_truth_table(x_data, w_or, "OR")
    
    # NAND
    y_nand = np.array([1, 1, 1, 0])
    w_nand = train_perceptron(x_data, y_nand, "NAND")
    
    color_idx = 0
    plt.figure(figsize=(8, 6))
    draw_learning(x_data, y_nand, w_nand, first_draw=True, title="NAND - результат")
    plt.legend()
    plt.show(block=True)
    
    print("Веса NAND:")
    print_weights(w_nand)
    print_truth_table(x_data, w_nand, "NAND")