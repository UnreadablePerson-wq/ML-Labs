# Лабораторная работа №3
# Обучение двухслойной ИНС для функции XOR
# Алгоритм обратного распространения ошибки (backpropagation)

import numpy as np
import math
import matplotlib.pyplot as plt

# Для воспроизводимости результатов
np.random.seed(42)

# Параметры обучения
LEARNING_RATE = 0.1  # скорость обучения (подобрал экспериментально)

# Веса сети
# У нас сеть: 2 входа + смещение -> скрытый слой (2 нейрона) -> выход (1 нейрон)
# Для скрытого слоя: каждый нейрон получает 3 входа (bias + x1 + x2)
weights_hidden = [
    np.random.uniform(-0.5, 0.5, 3),  # веса для первого скрытого нейрона
    np.random.uniform(-0.5, 0.5, 3)   # веса для второго скрытого нейрона
]

# Для выходного слоя: получает 3 входа (bias + выходы двух скрытых нейронов)
weights_output = np.random.uniform(-0.5, 0.5, 3)

# Обучающие данные для XOR
# Формат: [bias, x1, x2]
x_train = np.array([
    [1.0, 0, 0],
    [1.0, 0, 1],
    [1.0, 1, 0],
    [1.0, 1, 1]
])

# Правильные ответы (используем 0 и 1, потому что на выходе сигмоида)
y_train = np.array([0, 1, 1, 0])

def sigmoid(x):
    """
    Сигмоидальная функция активации
    Используется на выходном слое, выдает значения от 0 до 1
    """
    return 1.0 / (1.0 + math.exp(-x))

def sigmoid_derivative(output):
    """
    Производная сигмоиды
    Если сигмоида = 1/(1+e^(-x)), то производная = сигмоида * (1 - сигмоида)
    output - это уже вычисленное значение сигмоиды
    """
    return output * (1.0 - output)

def tanh_derivative(output):
    """
    Производная гиперболического тангенса
    Если tanh = (e^x - e^(-x))/(e^x + e^(-x)), 
    то производная = 1 - tanh^2
    output - это уже вычисленное значение tanh
    """
    return 1.0 - output**2

def show_learning(epoch):
    """
    Выводим текущие веса сети
    Вызывается после каждой эпохи
    """
    print(f"Эпоха {epoch:2d}: ", end="")
    # Веса скрытого слоя
    for i, w in enumerate(weights_hidden):
        print(f"w_h{i} = [", end="")
        for j, val in enumerate(w):
            print(f"{val:7.4f}", end=" ")
        print("] ", end="")
    # Веса выходного слоя
    print("w_o = [", end="")
    for val in weights_output:
        print(f"{val:7.4f}", end=" ")
    print("]")

def forward_pass(x):
    """
    Прямой проход по сети
    x - входной вектор [bias, x1, x2]
    Возвращает выходы всех нейронов
    """
    # Скрытый слой (tanh)
    hidden_outputs = []
    for neuron_weights in weights_hidden:
        # Сумма входов с весами
        z = np.dot(neuron_weights, x)
        # Активация tanh
        hidden_outputs.append(math.tanh(z))
    
    # Выходной слой (сигмоида)
    # Добавляем bias к выходам скрытого слоя
    output_inputs = np.array([1.0] + hidden_outputs)  # [bias, h1, h2]
    z_out = np.dot(weights_output, output_inputs)
    final_output = sigmoid(z_out)
    
    return hidden_outputs, final_output

def backward_pass(x, y, hidden_outputs, final_output):
    """
    Обратный проход - вычисляем ошибки для каждого нейрона
    x - входной вектор
    y - правильный ответ
    hidden_outputs - выходы скрытого слоя
    final_output - выход сети
    """
    # Ошибка на выходе
    # Производная функции ошибки: (y - output)
    output_error = y - final_output
    
    # Дельта выходного нейрона
    # умножаем на производную функции активации
    output_delta = output_error * sigmoid_derivative(final_output)
    
    # Ошибки скрытого слоя
    hidden_errors = []
    hidden_deltas = []
    
    for i in range(len(weights_hidden)):
        # Вклад этого нейрона в ошибку на выходе
        # умножаем дельту выхода на вес, связывающий этот нейрон с выходом
        hidden_error = output_delta * weights_output[i + 1]  # +1 потому что weights_output[0] - это bias
        hidden_errors.append(hidden_error)
        
        # Дельта скрытого нейрона
        hidden_delta = hidden_error * tanh_derivative(hidden_outputs[i])
        hidden_deltas.append(hidden_delta)
    
    return output_delta, hidden_deltas

def adjust_weights(x, y, hidden_outputs, final_output):
    """
    Корректируем веса на основе ошибок
    """
    global weights_hidden, weights_output
    
    # Сначала считаем все ошибки
    output_delta, hidden_deltas = backward_pass(x, y, hidden_outputs, final_output)
    
    # Корректируем веса выходного слоя
    # Входы для выходного слоя: [bias, h1, h2]
    output_inputs = np.array([1.0] + hidden_outputs)
    for j in range(len(weights_output)):
        weights_output[j] += LEARNING_RATE * output_delta * output_inputs[j]
    
    # Корректируем веса скрытого слоя
    for i in range(len(weights_hidden)):
        for j in range(len(weights_hidden[i])):
            weights_hidden[i][j] += LEARNING_RATE * hidden_deltas[i] * x[j]

def plot_training_history(errors):
    """
    Рисуем график изменения ошибки в процессе обучения
    """
    plt.figure(figsize=(10, 6))
    plt.plot(errors, 'b-', linewidth=2)
    plt.title('Изменение ошибки при обучении')
    plt.xlabel('Эпоха')
    plt.ylabel('Суммарная квадратичная ошибка')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # логарифмическая шкала лучше показывает изменение
    plt.show(block=True)

# Основной цикл обучения
print("Начинаем обучение двухслойной сети для XOR")
print("Структура: 2 входа -> 2 скрытых нейрона (tanh) -> 1 выход (сигмоида)")
print("Скорость обучения:", LEARNING_RATE)
print("\nНачальные веса:")
show_learning(0)

# Параметры обучения
max_epochs = 2000
all_correct = False
epoch = 0
indices = list(range(len(x_train)))

# Для графика ошибок
error_history = []

print("\nПроцесс обучения:")

while not all_correct and epoch < max_epochs:
    epoch += 1
    all_correct = True
    
    # Перемешиваем примеры
    np.random.shuffle(indices)
    
    # Суммарная ошибка за эпоху
    total_error = 0.0
    
    for i in indices:
        x = x_train[i]
        y = y_train[i]
        
        # Прямой проход
        hidden, output = forward_pass(x)
        
        # Считаем ошибку
        error = (y - output) ** 2  # квадратичная ошибка
        total_error += error
        
        # Проверяем, правильно ли сработала сеть
        # Округляем выход до 0 или 1 для проверки
        if abs(y - round(output)) > 0.1:  # допускаем небольшую погрешность
            all_correct = False
        
        # Корректируем веса
        adjust_weights(x, y, hidden, output)
    
    error_history.append(total_error)
    
    # Выводим результаты каждые 100 эпох
    if epoch % 100 == 0:
        show_learning(epoch)
        print(f"   Ошибка: {total_error:.6f}")

# Итоги обучения
print(f"\nОбучение завершено за {epoch} эпох")
print("\nФинальные веса:")
show_learning(epoch)

# Проверяем работу сети на всех примерах
print("\n" + "="*50)
print("ПРОВЕРКА РАБОТЫ СЕТИ")
print("="*50)
print(" x1  x2 | Правильно | Сеть выдала | Округленно | Статус")
print("-"*50)

correct_count = 0
for i in range(len(x_train)):
    x = x_train[i]
    y_true = y_train[i]
    _, output = forward_pass(x)
    rounded = round(output)
    
    if y_true == rounded:
        status = "+"
        correct_count += 1
    else:
        status = "-"
    
    print(f"  {int(x[1])}   {int(x[2])}  |     {y_true}     |   {output:.6f}   |     {rounded}     |   {status}")

print("-"*50)
print(f"Точность: {correct_count}/4 = {correct_count*25}%")

if correct_count == 4:
    print("Сеть успешно обучилась функции XOR!")
else:
    print("Сеть не смогла обучиться. Попробуйте изменить параметры.")

# Показываем график обучения
plot_training_history(error_history)

# Дополнительная визуализация - как менялась ошибка
plt.figure(figsize=(10, 6))
plt.plot(error_history[-200:], 'r-', linewidth=2)  # последние 200 эпох
plt.title('Ошибка на последних 200 эпохах')
plt.xlabel('Эпоха (относительно конца)')
plt.ylabel('Ошибка')
plt.grid(True, alpha=0.3)
plt.show(block=True)