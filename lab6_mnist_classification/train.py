import numpy as np
import matplotlib.pyplot as plt
import os
from data_loader import load_mnist, prepare_data, add_bias
from model import NeuralNetwork
from augmentation import augment_dataset
from visualize import plot_training_history, compare_models
import config

def train_epoch(model, x_train, y_train, batch_size=None):
    """
    Обучает модель одну эпоху.
    Если batch_size=None, используется стохастический градиентный спуск (по одному примеру)
    """
    indices = np.random.permutation(len(x_train))
    correct = 0
    
    for idx in indices:
        x = add_bias(x_train[idx])
        y_true = y_train[idx]
        
        output = model.train_step(x, y_true)
        
        if np.argmax(output) == np.argmax(y_true):
            correct += 1
    
    return correct / len(x_train)

def evaluate(model, x_test, y_test):
    """Оценка точности на тестовом наборе"""
    correct = 0
    for i in range(len(x_test)):
        x = add_bias(x_test[i])
        pred = model.predict(x)
        if pred == np.argmax(y_test[i]):
            correct += 1
    return correct / len(x_test)

def train_model(model, x_train, y_train, x_test, y_test, epochs, model_name=""):
    """
    Полный цикл обучения модели.
    Возвращает историю обучения и финальную точность.
    """
    history = {
        'train_acc': [],
        'test_acc': [],
        'train_error': [],
        'test_error': []
    }
    
    print(f"\n{'='*60}")
    print(f"ОБУЧЕНИЕ: {model_name}")
    print(f"{'='*60}")
    
    for epoch in range(epochs):
        train_acc = train_epoch(model, x_train, y_train)
        test_acc = evaluate(model, x_test, y_test)
        
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['train_error'].append(1 - train_acc)
        history['test_error'].append(1 - test_acc)
        
        print(f"Эпоха {epoch+1:2d}: train_acc = {train_acc:.4f}, test_acc = {test_acc:.4f}")
    
    return history, test_acc

def main():
    """Основная функция для запуска экспериментов"""
    
    # 1. Загрузка данных
    print("Загрузка данных...")
    train_imgs, train_lbls, test_imgs, test_lbls = load_mnist()
    x_train, y_train, x_test, y_test, mean, std = prepare_data(
        train_imgs, train_lbls, test_imgs, test_lbls
    )
    
    results = {}
    
    # 2. Модель 1: Базовая
    print("\n" + "="*60)
    print("ЭКСПЕРИМЕНТ 1: Базовая модель")
    print("="*60)
    
    model1 = NeuralNetwork(
        hidden_size=config.BASE_HIDDEN,
        learning_rate=config.BASE_LR
    )
    hist1, acc1 = train_model(
        model1, x_train, y_train, x_test, y_test,
        epochs=config.BASE_EPOCHS,
        model_name="Базовая"
    )
    results['Базовая'] = {'history': hist1, 'accuracy': acc1}
    
    # Сохраняем веса
    os.makedirs('results', exist_ok=True)
    model1.save_weights('results/model_base.npz')
    
    # 3. Модель 2: Улучшенная
    print("\n" + "="*60)
    print("ЭКСПЕРИМЕНТ 2: Улучшенная модель")
    print("="*60)
    
    model2 = NeuralNetwork(
        hidden_size=config.BEST_HIDDEN,
        learning_rate=config.BEST_LR
    )
    hist2, acc2 = train_model(
        model2, x_train, y_train, x_test, y_test,
        epochs=config.BEST_EPOCHS,
        model_name="Улучшенная"
    )
    results['Улучшенная'] = {'history': hist2, 'accuracy': acc2}
    model2.save_weights('results/model_best.npz')
    
    # 4. Модель 3: С аугментацией
    print("\n" + "="*60)
    print("ЭКСПЕРИМЕНТ 3: Модель с аугментацией")
    print("="*60)
    
    print("Создание расширенного набора данных...")
    x_aug, y_aug = augment_dataset(train_imgs, train_lbls, mean, std)
    print(f"Размер расширенного набора: {len(x_aug)} примеров")
    
    model3 = NeuralNetwork(
        hidden_size=config.BEST_HIDDEN,
        learning_rate=config.BEST_LR
    )
    hist3, acc3 = train_model(
        model3, x_aug, y_aug, x_test, y_test,
        epochs=config.BEST_EPOCHS,
        model_name="С аугментацией"
    )
    results['С аугментацией'] = {'history': hist3, 'accuracy': acc3}
    model3.save_weights('results/model_augmented.npz')
    
    # 5. Визуализация и сравнение
    plot_training_history(results)
    compare_models(results)
    
    # 6. Итоговый вывод
    print("\n" + "="*60)
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("="*60)
    for name, res in results.items():
        print(f"{name:20s}: {res['accuracy']:.4%}")
    
    return results

if __name__ == "__main__":
    main()