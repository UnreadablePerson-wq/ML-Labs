import matplotlib.pyplot as plt
import numpy as np
import os

def plot_training_history(results):
    """
    Строит графики обучения для всех моделей.
    results: словарь вида {'имя_модели': {'history': {...}, 'accuracy': ...}}
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for name, data in results.items():
        hist = data['history']
        epochs = range(1, len(hist['train_acc']) + 1)
        
        # График точности
        axes[0].plot(epochs, hist['train_acc'], '--', label=f'{name} (train)')
        axes[0].plot(epochs, hist['test_acc'], '-', label=f'{name} (test)')
        
        # График ошибки
        axes[1].plot(epochs, hist['train_error'], '--', label=f'{name} (train)')
        axes[1].plot(epochs, hist['test_error'], '-', label=f'{name} (test)')
    
    axes[0].set_xlabel('Эпохи')
    axes[0].set_ylabel('Точность')
    axes[0].set_title('Динамика точности')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].set_xlabel('Эпохи')
    axes[1].set_ylabel('Ошибка (1 - точность)')
    axes[1].set_title('Динамика ошибки')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('results/training_history.png', dpi=150)
    plt.show()

def compare_models(results):
    """Строит сравнительную столбчатую диаграмму"""
    names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in names]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    
    # Добавляем значения на столбцы
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2%}',
                ha='center', va='bottom', fontsize=12)
    
    plt.ylabel('Точность на тесте')
    plt.title('Сравнение точности моделей')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/model_comparison.png', dpi=150)
    plt.show()

def show_misclassified(model, x_test, y_test, test_images, num_examples=10):
    """
    Показывает примеры изображений, на которых модель ошибается.
    """
    misclassified = []
    
    for i in range(len(x_test)):
        x = np.concatenate(([1.0], x_test[i]))
        pred = model.predict(x)
        true = np.argmax(y_test[i])
        if pred != true:
            misclassified.append((i, true, pred))
            if len(misclassified) >= num_examples:
                break
    
    if not misclassified:
        print("Ошибочных классификаций не найдено!")
        return
    
    cols = 5
    rows = (len(misclassified) + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3*rows))
    axes = axes.flatten() if rows > 1 else [axes]
    
    for idx, (ax, (img_idx, true, pred)) in enumerate(zip(axes, misclassified)):
        img = test_images[img_idx]
        ax.imshow(img, cmap='gray')
        ax.set_title(f'Истина: {true}\nПредсказание: {pred}', color='red')
        ax.axis('off')
    
    # Скрыть лишние оси
    for ax in axes[len(misclassified):]:
        ax.axis('off')
    
    plt.suptitle('Примеры ошибочной классификации', fontsize=16)
    plt.tight_layout()
    plt.savefig('results/misclassified.png', dpi=150)
    plt.show()