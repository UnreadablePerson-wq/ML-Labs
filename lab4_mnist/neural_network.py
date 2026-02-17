# Реализация двухслойной нейронной сети с нуля
# Вход: 784 пикселя
# Скрытый слой: 64 нейрона с tanh
# Выход: 1 нейрон с сигмоидой

import numpy as np

class NeuralNetwork:
    """
    Двухслойная нейронная сеть для бинарной классификации
    """
    
    def __init__(self, input_size=784, hidden_size=64, learning_rate=0.01):
        """
        Инициализация сети
        - input_size: количество входных нейронов (пикселей)
        - hidden_size: размер скрытого слоя
        - learning_rate: скорость обучения
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        
        # Инициализация весов (важно выбрать правильный масштаб)
        np.random.seed(42)  # для воспроизводимости
        
        # Веса от входа к скрытому слою
        # Умножаем на 0.1, чтобы не было слишком больших значений
        self.w1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros(hidden_size)
        
        # Веса от скрытого слоя к выходу
        self.w2 = np.random.randn(hidden_size, 1) * 0.1
        self.b2 = np.zeros(1)
        
        # Для отслеживания ошибки
        self.loss_history = []
        
    def sigmoid(self, x):
        """
        Сигмоидальная функция активации
        Используется на выходном слое
        """
        # Ограничиваем x, чтобы избежать переполнения
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """
        Производная сигмоиды
        x - это уже выход сигмоиды (для эффективности)
        """
        return x * (1.0 - x)
    
    def tanh(self, x):
        """
        Гиперболический тангенс
        Используется на скрытом слое
        """
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        """
        Производная tanh
        x - это уже выход tanh
        """
        return 1.0 - x**2
    
    def forward(self, X):
        """
        Прямой проход по сети
        X - входные данные (может быть один пример или батч)
        Возвращает выход сети и промежуточные значения
        """
        # Приводим X к правильной форме, если это один пример
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Скрытый слой
        self.z1 = np.dot(X, self.w1) + self.b1
        self.a1 = self.tanh(self.z1)
        
        # Выходной слой
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        
        return self.a2
    
    def backward(self, X, y, output):
        """
        Обратное распространение ошибки
        X - входные данные
        y - правильные ответы
        output - выход сети
        """
        # Приводим к правильной форме
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Ошибка на выходе (производная MSE)
        d_output = 2 * (output - y.reshape(-1, 1)) * self.sigmoid_derivative(output)
        
        # Ошибка на скрытом слое
        d_hidden = np.dot(d_output, self.w2.T) * self.tanh_derivative(self.a1)
        
        # Градиенты для выходного слоя
        grad_w2 = np.dot(self.a1.T, d_output)
        grad_b2 = d_output.sum(axis=0)
        
        # Градиенты для скрытого слоя
        grad_w1 = np.dot(X.T, d_hidden)
        grad_b1 = d_hidden.sum(axis=0)
        
        return grad_w1, grad_b1, grad_w2, grad_b2
    
    def update_weights(self, grad_w1, grad_b1, grad_w2, grad_b2):
        """
        Обновление весов по градиентам
        """
        self.w2 -= self.learning_rate * grad_w2
        self.b2 -= self.learning_rate * grad_b2
        self.w1 -= self.learning_rate * grad_w1
        self.b1 -= self.learning_rate * grad_b1
    
    def train_step(self, X, y):
        """
        Один шаг обучения на одном примере
        """
        # Прямой проход
        output = self.forward(X)
        
        # Считаем ошибку
        loss = ((output - y) ** 2).mean()
        
        # Обратный проход
        grad_w1, grad_b1, grad_w2, grad_b2 = self.backward(X, y, output)
        
        # Обновляем веса
        self.update_weights(grad_w1, grad_b1, grad_w2, grad_b2)
        
        return loss
    
    def predict(self, X, threshold=0.5):
        """
        Предсказание класса
        Возвращает 1 (цифра 8) или 0 (не цифра 8)
        """
        output = self.forward(X)
        return (output > threshold).astype(int)
    
    def save_weights(self, filename):
        """Сохраняет веса в файл"""
        np.savez(filename, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2)
    
    def load_weights(self, filename):
        """Загружает веса из файла"""
        data = np.load(filename)
        self.w1 = data['w1']
        self.b1 = data['b1']
        self.w2 = data['w2']
        self.b2 = data['b2']


def test_network():
    """
    Простой тест работы сети
    """
    print("Тестирование сети...")
    
    # Создаем сеть
    nn = NeuralNetwork(input_size=784, hidden_size=64, learning_rate=0.01)
    
    # Случайный тестовый пример
    X_test = np.random.rand(784)
    y_test = np.array([1])
    
    # Прямой проход
    output = nn.forward(X_test)
    print(f"Выход сети на случайном входе: {output[0,0]:.6f}")
    
    # Шаг обучения
    loss = nn.train_step(X_test, y_test)
    print(f"Ошибка после одного шага: {loss:.6f}")
    
    # Предсказание
    pred = nn.predict(X_test)
    print(f"Предсказание: {pred[0,0]}")
    
    return nn

if __name__ == "__main__":
    test_network()