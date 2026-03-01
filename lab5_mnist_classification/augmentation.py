# augmentation.py
import numpy as np
from scipy.ndimage import shift

def shift_image(image, direction):
    """
    Сдвигает изображение 28x28 на один пиксель.
    direction: 'left', 'right', 'up', 'down'
    """
    shifts = {
        'left': (0, -1),
        'right': (0, 1),
        'up': (-1, 0),
        'down': (1, 0)
    }
    
    if direction not in shifts:
        raise ValueError(f"Unknown direction: {direction}")
    
    return shift(image, shifts[direction], mode='constant', cval=0)

def augment_dataset(images, labels, mean, std):
    """
    Создает расширенный набор данных (оригинал + 4 сдвига).
    images: исходные изображения (60000, 28, 28) со значениями 0-255
    labels: исходные метки
    mean, std: статистики для нормализации
    """
    n = len(images)
    x_aug_list = []
    y_aug_list = []
    
    print(f"Создание расширенного набора из {n} изображений...")
    
    for i in range(n):
        img = images[i]
        label = labels[i]
        
        # Оригинал
        x_aug_list.append(img.reshape(784))
        y_aug_list.append(label)
        
        # Сдвиги
        for direction in ['left', 'right', 'up', 'down']:
            shifted = shift_image(img, direction)
            x_aug_list.append(shifted.reshape(784))
            y_aug_list.append(label)
        
        if (i + 1) % 10000 == 0:
            print(f"Обработано {i + 1}/{n} изображений...")
    
    print("Преобразование в массивы numpy...")
    # Преобразование в массивы
    x_aug_raw = np.array(x_aug_list)
    y_aug = np.zeros((len(x_aug_raw), 10))
    for i, lab in enumerate(y_aug_list):
        y_aug[i, lab] = 1
    
    print("Нормализация данных...")
    # Нормализация
    x_aug = (x_aug_raw - mean) / std
    
    print(f"Расширенный набор создан! Размер: {len(x_aug)} примеров")
    
    return x_aug, y_aug