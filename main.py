import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Загрузка данных Iris
iris = load_iris()
X = iris.data
y = iris.target

# Ограничим данные только двумя классами (0 и 1)
X = X[y != 2]
y = y[y != 2]

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Нормализация данных
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Создание модели
model = models.Sequential()

# Полносвязный слой с 10 нейронами
model.add(layers.Dense(10, activation='relu', input_shape=(X_train.shape[1],)))

# Выходной слой с 1 нейроном (для бинарной классификации)
model.add(layers.Dense(1, activation='sigmoid'))

# Компиляция модели
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Обучение модели
model.fit(X_train, y_train, epochs=10, batch_size=10, validation_data=(X_test, y_test))

# Оценка модели
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Точность на тестовых данных: {test_acc}')