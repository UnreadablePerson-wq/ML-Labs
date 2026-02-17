# Визуализация разделяющих поверхностей
# Показываем, как сеть разделяет пространство входов

import numpy as np
import matplotlib.pyplot as plt
from xor_backprop import forward_pass, x_train, y_train
import xor_backprop

def plot_decision_boundary(resolution=0.05):
    """
    Рисуем, как сеть классифицирует точки на плоскости
    """
    # Создаем сетку точек
    x1 = np.arange(-0.2, 1.2, resolution)
    x2 = np.arange(-0.2, 1.2, resolution)
    X1, X2 = np.meshgrid(x1, x2)
    
    # Для каждой точки считаем выход сети
    Z = np.zeros(X1.shape)
    for i in range(len(x1)):
        for j in range(len(x2)):
            _, output = forward_pass(np.array([1.0, X1[i, j], X2[i, j]]))
            Z[i, j] = output
    
    plt.figure(figsize=(10, 8))
    
    # Рисуем цветовую карту
    plt.contourf(X1, X2, Z, levels=20, cmap='RdBu', alpha=0.7)
    plt.colorbar(label='Выход сети')
    
    # Наносим точки обучающей выборки
    for i in range(len(x_train)):
        x1_val = x_train[i][1]
        x2_val = x_train[i][2]
        if y_train[i] == 1:
            plt.plot(x1_val, x2_val, 'ro', markersize=10, label='Класс 1' if i==0 else '')
        else:
            plt.plot(x1_val, x2_val, 'bo', markersize=10, label='Класс 0' if i==0 else '')
    
    plt.title('Разделяющая поверхность, построенная сетью')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim(-0.2, 1.2)
    plt.ylim(-0.2, 1.2)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show(block=True)

def plot_network_structure():
    """
    Рисуем схему сети для отчета
    """
    plt.figure(figsize=(12, 8))
    plt.title('Структура двухслойной сети для XOR', fontsize=14)
    plt.axis('off')
    
    # Позиции нейронов
    input_pos = [(2, 5), (2, 4), (2, 3)]  # bias, x1, x2
    hidden_pos = [(6, 5), (6, 3)]  # два скрытых нейрона
    output_pos = [(10, 4)]  # выходной нейрон
    
    # Рисуем входной слой
    plt.text(1, 5.2, 'Входной слой', fontsize=12)
    for i, (x, y) in enumerate(input_pos):
        if i == 0:
            label = 'bias = 1'
        else:
            label = f'x{i}'
        circle = plt.Circle((x, y), 0.3, fill=False, linewidth=2)
        plt.gca().add_patch(circle)
        plt.text(x, y-0.5, label, ha='center')
    
    # Рисуем скрытый слой
    plt.text(6, 5.7, 'Скрытый слой (tanh)', fontsize=12)
    for i, (x, y) in enumerate(hidden_pos):
        circle = plt.Circle((x, y), 0.3, fill=False, linewidth=2)
        plt.gca().add_patch(circle)
        plt.text(x, y-0.7, f'h{i+1}', ha='center')
    
    # Рисуем выходной слой
    plt.text(10, 4.7, 'Выходной слой (сигмоида)', fontsize=12)
    circle = plt.Circle(output_pos[0], 0.3, fill=False, linewidth=2)
    plt.gca().add_patch(circle)
    plt.text(output_pos[0][0], output_pos[0][1]-0.7, 'y', ha='center')
    
    # Рисуем связи
    for inp in input_pos:
        for hid in hidden_pos:
            plt.plot([inp[0]+0.3, hid[0]-0.3], [inp[1], hid[1]], 'gray', linewidth=1, alpha=0.5)
    
    for hid in hidden_pos:
        plt.plot([hid[0]+0.3, output_pos[0][0]-0.3], [hid[1], output_pos[0][1]], 'gray', linewidth=1, alpha=0.5)
    
    plt.xlim(0, 12)
    plt.ylim(2, 6.5)
    plt.show(block=True)

if __name__ == "__main__":
    print("Визуализация результатов обучения")
    print("\n1. Разделяющая поверхность")
    plot_decision_boundary()
    
    print("\n2. Структура сети")
    plot_network_structure()