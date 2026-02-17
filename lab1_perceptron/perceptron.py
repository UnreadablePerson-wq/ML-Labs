# Лабораторная работа №1
# Алгоритм обучения персептрона
# Вариант 18

import matplotlib.pyplot as plt
import random
from variant18_data import x_train, y_train

# Настройка случайности для воспроизводимости
random.seed(7)

# Параметры обучения
LEARNING_RATE = 0.1  # скорость обучения
max_epochs = 100     # на всякий случай ограничим

# Начальные веса (взял небольшие, чтобы обучение не скакало)
w = [0.2, -0.3, 0.1]  # w0 - bias, w1 - для x2, w2 - для x1

# Цвета для линий на разных эпохах
colors = ['r-', 'black', 'y-', 'c-', 'b-', 'g-']
current_color = 0

def compute_output(weights, inputs):
    """
    Считаем взвешенную сумму и применяем пороговую функцию
    """
    z = 0.0
    for i in range(len(weights)):
        z += inputs[i] * weights[i]
    
    # Возвращаем -1 или 1 в зависимости от знака
    if z < 0:
        return -1
    else:
        return 1

def plot_learning(weights, first_plot=False):
    """
    Рисуем текущую разделяющую линию
    first_plot - если True, то сначала рисуем все точки
    """
    global current_color
    
    # Выводим веса в консоль
    print(f'w0 = {weights[0]:6.3f}, w1 = {weights[1]:6.3f}, w2 = {weights[2]:6.3f}')
    
    # Точки для построения линии (по x1 от -2 до 2)
    x1_line = [-2, 2]
    
    # Из уравнения w0 + w1*x2 + w2*x1 = 0 выражаем x2:
    # w1*x2 = -w0 - w2*x1
    # x2 = (-w0 - w2*x1) / w1
    
    if abs(weights[1]) < 1e-5:  # защита от деления на ноль
        x2_line = [0, 0]
    else:
        x2_line = [(-weights[0] - weights[2]*x1_line[0]) / weights[1],
                   (-weights[0] - weights[2]*x1_line[1]) / weights[1]]
    
    if first_plot:
        # Рисуем точки обучающей выборки
        for i in range(len(x_train)):
            x1 = x_train[i][2]  # x1
            x2 = x_train[i][1]  # x2
            if y_train[i] == -1:
                plt.plot(x1, x2, 'b_', markersize=12)  # синие для -1
            else:
                plt.plot(x1, x2, 'r+', markersize=12)  # красные для +1
        
        # Настройки графика
        plt.axis([-2, 2, -2, 2])
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Процесс обучения персептрона')
        plt.grid(True, alpha=0.3)
    
    # Рисуем линию
    plt.plot(x1_line, x2_line, colors[current_color], linewidth=2, 
             label=f'Эпоха {current_color+1}' if current_color < 5 else '')
    
    # Меняем цвет
    current_color = (current_color + 1) % len(colors)
    
    # Пауза, чтобы увидеть изменения
    plt.pause(0.5)

def plot_final(weights):
    """
    Рисуем финальный график только с одной линией
    """
    plt.figure()  # создаем новое окно
    
    # Рисуем точки
    for i in range(len(x_train)):
        x1 = x_train[i][2]
        x2 = x_train[i][1]
        if y_train[i] == -1:
            plt.plot(x1, x2, 'b_', markersize=12)
        else:
            plt.plot(x1, x2, 'r+', markersize=12)
    
    # Рисуем финальную линию
    x1_line = [-2, 2]
    if abs(weights[1]) < 1e-5:
        x2_line = [0, 0]
    else:
        x2_line = [(-weights[0] - weights[2]*x1_line[0]) / weights[1],
                   (-weights[0] - weights[2]*x1_line[1]) / weights[1]]
    
    plt.plot(x1_line, x2_line, 'g-', linewidth=3, label='Финальная линия')
    
    # Настройки
    plt.axis([-2, 2, -2, 2])
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('Конечный результат обучения')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

def train_perceptron():
    global w, current_color
    
    print("Начинаем обучение")
    print(f"Всего примеров: {len(x_train)}")
    print("\nНачальные веса:")
    plot_learning(w, first_plot=True)
    
    # Индексы для перемешивания
    indices = list(range(len(x_train)))
    
    all_correct = False
    epoch = 0
    
    print("\nПроцесс обучения:")
    
    while not all_correct and epoch < max_epochs:
        epoch += 1
        all_correct = True
        
        # Перемешиваем примеры
        random.shuffle(indices)
        
        # Проходим по всем примерам
        for idx in indices:
            x = x_train[idx]
            target = y_train[idx]
            
            output = compute_output(w, x)
            
            # Если ошибка - корректируем веса
            if output != target:
                for j in range(len(w)):
                    w[j] += LEARNING_RATE * target * x[j]
                all_correct = False
        
        # После каждой эпохи показываем новую линию
        print(f"Эпоха {epoch:2d}: ", end="")
        plot_learning(w)
    
    # Итог
    if all_correct:
        print(f"\n✅ Готово! Понадобилось {epoch} эпох")
    else:
        print(f"\n⚠️ Стоп, достигли максимума ({max_epochs} эпох)")
    
    print("\nФинальные веса:")
    print(f'w0 = {w[0]:.3f}, w1 = {w[1]:.3f}, w2 = {w[2]:.3f}')
    
    # Показываем первый график с процессом обучения
    plt.legend(loc='upper right')
    plt.show()
    
    # Показываем второй график с финальным результатом
    print("\nФинальный график:")
    plot_final(w)

if __name__ == "__main__":
    train_perceptron()