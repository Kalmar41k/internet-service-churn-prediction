import streamlit as st
import pickle
import tensorflow as tf
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Завантаження даних для тренування моделі
X_train = pd.read_csv("data/X_train.csv")

# Стандартизація даних
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Шлях до моделей
models = {
    "Logistic Regression": "models/logistic_regression_model.pkl", # Logistic regression
    "Random Forest": "models/random_forest_model.pkl", # Random forest classifier
    "SVM": "models/best_svm_model.pkl", # Support vector machine
    "Neural Network": "models/dnn_model.h5" # Deep neural network
}

# Завантаження моделей
loaded_models = {}
for name, path in models.items():
    if name == "Neural Network": # Якщо модель — нейронна мережа
        loaded_models[name] = tf.keras.models.load_model(path) # Завантажуємо модель через TensorFlow.
    elif name == "SVM": # Якщо модель — SVM.
        loaded_models[name] = joblib.load(path) # Завантажуємо модель через joblib.
    else: # Для інших моделей (Logistic Regression, Random Forest).
        with open(path, "rb") as f:
            loaded_models[name] = pickle.load(f) # Завантажуємо модель через pickle.

# Інтерфейс Streamlit
st.title("Прогнозування відтоку клієнтів") # Заголовок для веб-додатку.
selected_model = st.selectbox("Оберіть модель", list(models.keys())) # Дропдаун для вибору моделі.

# Поля для введення значень ознак
st.subheader("Введіть значення ознак") # Підзаголовок.
user_input = {} # Словник для збереження введених значень.
for column in X_train.columns: # Беремо мінімальне та максимальне значення кожного стовпця
    min_val = float(X_train[column].min())
    max_val = float(X_train[column].max())
    default_val = (min_val + max_val) / 2  # За замовчуванням середнє значення між мінімумом і максимумом
    user_input[column] = st.number_input( # Встановлюємо min та max межі для введення значень
        f"{column}", 
        min_value=min_val, 
        max_value=max_val, 
        value=default_val
    )

# Кнопка для запуску прогнозування
if st.button("Зробити прогноз"): # Якщо користувач натиснув кнопку "Зробити прогноз".
    input_data = pd.DataFrame([user_input], columns=X_train.columns) # Перетворюємо введені значення в DataFrame.

    # Для моделей, які не потребують нормалізації (наприклад, Random Forest)
    if selected_model == "Random Forest":
        transformed_data = input_data # Використовуємо введені дані без масштабування.
    else:
        transformed_data = scaler.transform(input_data) # Масштабуємо введені дані.

    model = loaded_models[selected_model] # Завантажуємо обрану модель.

    # Для нейронної мережі, прогнозуємо ймовірність і клас
    if selected_model == "Neural Network":
        prediction = model.predict(transformed_data)[0, 0] # Отримуємо прогноз для нейронної мережі.
        predicted_class = int(prediction > 0.5) # Перетворюємо прогноз у клас (1 або 0).
        probability = prediction if predicted_class == 1 else (1 - prediction) # Обчислюємо ймовірність.
    else:
        predicted_class = model.predict(transformed_data)[0] # Для інших моделей передбачаємо клас.
        probability = model.predict_proba(transformed_data)[0][predicted_class] # Обчислюємо ймовірність для класу.

    # Виведення результату
    st.write(
        f"**Результат:** {'Залишається' if predicted_class == 0 else 'Йде'}")
    if predicted_class == 0:
        st.write(f"**Ймовірність, що клієнт залишиться:** {probability:.2%}")
    else:
        st.write(f"**Ймовірність, що клієнт піде:** {probability:.2%}")