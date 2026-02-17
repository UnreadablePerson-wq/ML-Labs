# Основная программа для варианта 18
# Схема: (x1 AND x2) AND (x1 XOR x3)

import numpy as np
import matplotlib.pyplot as plt
from logic_gates import compute_output, train_perceptron, draw_learning
import logic_gates

print("=" * 60)
print("Лабораторная работа №2: Нейронная сеть на персептронах")
print("Вариант 18: (x1 AND x2) AND (x1 XOR x3)")
print("=" * 60)

# Обучаем все нужные элементы
print("\nПодготовка персептронов...")

# Данные для обучения
x_data = np.array([
    [1.0, 0, 0],
    [1.0, 0, 1],
    [1.0, 1, 0],
    [1.0, 1, 1]
])

# AND
y_and = np.array([0, 0, 0, 1])
w_and = train_perceptron(x_data, y_and, "AND (для сети)")

# OR
y_or = np.array([0, 1, 1, 1])
w_or = train_perceptron(x_data, y_or, "OR (для сети)")

# NAND
y_nand = np.array([1, 1, 1, 0])
w_nand = train_perceptron(x_data, y_nand, "NAND (для сети)")

print("\nВсе персептроны обучены!")

def xor_with_perceptrons(a, b):
    """
    Реализуем XOR через OR и NAND
    Формула: XOR = (a OR b) AND (a NAND b)
    """
    a_or_b = compute_output(w_or, np.array([1.0, a, b]))
    a_nand_b = compute_output(w_nand, np.array([1.0, a, b]))
    return compute_output(w_and, np.array([1.0, a_or_b, a_nand_b]))

def neural_network(x1, x2, x3):
    """
    Полная схема: (x1 AND x2) AND (x1 XOR x3)
    Возвращает: (результат, промежуточные значения)
    """
    # Первый слой
    y1 = compute_output(w_and, np.array([1.0, x1, x2]))  # x1 AND x2
    
    # Второй слой - вычисляем XOR через OR и NAND
    x1_or_x3 = compute_output(w_or, np.array([1.0, x1, x3]))
    x1_nand_x3 = compute_output(w_nand, np.array([1.0, x1, x3]))
    y2 = compute_output(w_and, np.array([1.0, x1_or_x3, x1_nand_x3]))  # x1 XOR x3
    
    # Выходной слой
    result = compute_output(w_and, np.array([1.0, y1, y2]))  # y1 AND y2
    
    return result, y1, y2, x1_or_x3, x1_nand_x3

# Тестирование
print("\n" + "=" * 70)
print("ТЕСТИРОВАНИЕ СХЕМЫ")
print("=" * 70)
print("x1 x2 x3 | x1ANDx2  x1ORx3  x1NANDx3 | x1XORx3 |  F  | Статус")
print("-" * 60)

# Все возможные комбинации входов
test_inputs = [
    [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
    [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]
]

# Правильные ответы из таблицы истинности
expected = [0, 0, 0, 0, 0, 0, 1, 0]
correct = 0

for i, (x1, x2, x3) in enumerate(test_inputs):
    f, y1, y2, or_out, nand_out = neural_network(x1, x2, x3)
    
    status = "+" if f == expected[i] else "-"
    if f == expected[i]:
        correct += 1
    
    print(f" {x1}  {x2}  {x3}  |    {y1}        {or_out}         {nand_out}     |     {y2}     |  {f}  |   {status}")

print("-" * 60)
print(f"Точность: {correct}/8 = {correct/8*100:.1f}%")

if correct == 8:
    print("Все тесты пройдены!")
else:
    print("Есть ошибки")

print("\n" + "=" * 50)
print("ОБУЧЕННЫЕ ВЕСА")
print("=" * 50)
print(f"AND:  w0 = {w_and[0]:.4f} (смещение), w1 = {w_and[1]:.4f}, w2 = {w_and[2]:.4f}")
print(f"OR:   w0 = {w_or[0]:.4f} (смещение), w1 = {w_or[1]:.4f}, w2 = {w_or[2]:.4f}")
print(f"NAND: w0 = {w_nand[0]:.4f} (смещение), w1 = {w_nand[1]:.4f}, w2 = {w_nand[2]:.4f}")

print("\n" + "=" * 60)
print("СХЕМА РАБОТЫ")
print("=" * 60)
print("1. Считаем x1 AND x2")
print("2. Считаем x1 XOR x3 через OR и NAND:")
print("   - x1 OR x3")
print("   - x1 NAND x3")
print("   - AND от этих двух результатов")
print("3. Итоговый AND от (x1ANDx2) и (x1XORx3)")

plt.show(block=True)