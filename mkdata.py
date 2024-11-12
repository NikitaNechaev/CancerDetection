import numpy as np
import pandas as pd
import cv2 as cv
import csv, os
from tqdm import tqdm

folder = 'C:\\Code\\Melanoma_detection\\Dataset\\Dataset\\Small_dir'
files = []
colors = []

for entry in os.scandir(folder):
    if entry.is_file():
        files.append(entry.name)

#for i in tqdm(files):


colors.append(cv.imread(f"{folder}\\{files[0]}")[0])

arr = np.array(cv.imread(f"{folder}\\{files[0]}"))

print(arr.shape)







from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

tf.keras.backend.clear_session()

from tensorflow import keras
from keras import layers

inputs = keras.Input(shape=(784,), name='digits')
x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
x = layers.Dense(64, activation='relu', name='dense_2')(x)
outputs = layers.Dense(10, activation='softmax', name='predictions')(x)

model = keras.Model(inputs=inputs, outputs=outputs)

# Загрузим учебный датасет для этого примера
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print(x_train.shape)

# Предобработаем данные (это массивы Numpy)
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255

y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

# Зарезервируем 10,000 примеров для валидации
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

# Укажем конфигурацию обучения (оптимизатор, функция потерь, метрики)
model.compile(optimizer=keras.optimizers.RMSprop(),  # Optimizer
              # Минимизируемая функция потерь
              loss=keras.losses.SparseCategoricalCrossentropy(),
              # Список метрик для мониторинга
              metrics=[keras.metrics.SparseCategoricalAccuracy()])

# Обучим модель разбив данные на "пакеты"
# размером "batch_size", и последовательно итерируя
# весь датасет заданное количество "эпох"
print('# Обучаем модель на тестовых данных')
history = model.fit(x_train, y_train,
                    batch_size=64,
                    epochs=3,
                    # Мы передаем валидационные данные для
                    # мониторинга потерь и метрик на этих данных
                    # в конце каждой эпохи
                    validation_data=(x_val, y_val))

# Возвращаемый объект "history" содержит записи
# значений потерь и метрик во время обучения
print('\nhistory dict:', history.history)

# Оценим модель на тестовых данных, используя "evaluate"
print('\n# Оцениваем на тестовых данных')
results = model.evaluate(x_test, y_test, batch_size=128)
print('test loss, test acc:', results)

# Сгенерируем прогнозы (вероятности - выходные данные последнего слоя)
# на новых данных с помощью "predict"
print('\n# Генерируем прогнозы для 3 образцов')
predictions = model.predict(x_test[:3])
print('размерность прогнозов:', predictions.shape)