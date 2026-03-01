import numpy as np

class NeuralNetwork:
    """
    Двухслойная нейронная сеть для классификации MNIST.
    Скрытый слой: tanh, выходной слой: sigmoid.
    """
    
    def __init__(self, input_size=784, hidden_size=64, output_size=10, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = learning_rate
        
        # Инициализация весов
        self.hidden_w = self._init_weights(hidden_size, input_size)
        self.output_w = self._init_weights(output_size, hidden_size)
        
        # Для хранения промежуточных значений при forward/backward
        self.hidden_y = None
        self.output_y = None
        self.hidden_error = None
        self.output_error = None
    
    def _init_weights(self, neurons, inputs):
        """Инициализация весов слоя"""
        weights = np.zeros((neurons, inputs + 1))  # +1 для bias
        for i in range(neurons):
            for j in range(1, inputs + 1):
                weights[i, j] = np.random.uniform(-0.1, 0.1)
        return weights
    
    def _tanh(self, x):
        return np.tanh(x)
    
    def _tanh_deriv(self, y):
        return 1.0 - y**2
    
    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))  # защита от переполнения
    
    def _sigmoid_deriv(self, y):
        return y * (1.0 - y)
    
    def forward(self, x):
        """
        Прямой проход.
        x: входной вектор с bias (длина input_size + 1)
        """
        # Скрытый слой
        z_hidden = np.dot(self.hidden_w, x)
        self.hidden_y = self._tanh(z_hidden)
        
        # Добавляем bias для выходного слоя
        hidden_with_bias = np.concatenate(([1.0], self.hidden_y))
        
        # Выходной слой
        z_output = np.dot(self.output_w, hidden_with_bias)
        self.output_y = self._sigmoid(z_output)
        
        return self.output_y
    
    def backward(self, y_true):
        """
        Обратный проход.
        y_true: правильный one-hot вектор
        """
        # Ошибка выходного слоя
        error_prime = -(y_true - self.output_y)  # производная MSE
        deriv_output = self._sigmoid_deriv(self.output_y)
        self.output_error = error_prime * deriv_output
        
        # Ошибка скрытого слоя
        # Веса от скрытого к выходному (без bias'а выходного слоя)
        weights_to_hidden = self.output_w[:, 1:]  # пропускаем первый столбец (bias)
        weighted_error = np.dot(weights_to_hidden.T, self.output_error)
        deriv_hidden = self._tanh_deriv(self.hidden_y)
        self.hidden_error = weighted_error * deriv_hidden
    
    def update_weights(self, x):
        """
        Обновление весов.
        x: входной вектор с bias
        """
        # Обновление весов скрытого слоя
        for i in range(self.hidden_size):
            self.hidden_w[i] -= self.lr * self.hidden_error[i] * x
        
        # Обновление весов выходного слоя
        hidden_with_bias = np.concatenate(([1.0], self.hidden_y))
        for i in range(self.output_size):
            self.output_w[i] -= self.lr * self.output_error[i] * hidden_with_bias
    
    def train_step(self, x, y_true):
        """Один шаг обучения на одном примере"""
        self.forward(x)
        self.backward(y_true)
        self.update_weights(x)
        return self.output_y
    
    def predict(self, x):
        """Предсказание класса для одного примера"""
        self.forward(x)
        return np.argmax(self.output_y)
    
    def save_weights(self, filepath):
        """Сохранение весов"""
        np.savez(filepath, hidden_w=self.hidden_w, output_w=self.output_w)
    
    def load_weights(self, filepath):
        """Загрузка весов"""
        data = np.load(filepath)
        self.hidden_w = data['hidden_w']
        self.output_w = data['output_w']