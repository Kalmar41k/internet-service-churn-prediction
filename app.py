import streamlit as st
import pickle
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

X_train = pd.read_csv("X_train.csv")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

models = {
    "Logistic Regression": "models/logistic_regression.pkl",
    "Random Forest": "models/random_forest.pkl",
    "Gradient Boosting": "models/gradient_boosting.pkl",
    "Neural Network": "models/neural_network.h5"
}

loaded_models = {}
for name, path in models.items():
    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            loaded_models[name] = pickle.load(f)
    elif path.endswith(".h5"):
        loaded_models[name] = tf.keras.models.load_model(path)

st.title("Прогнозування відтоку клієнтів")
selected_model = st.selectbox("Оберіть модель", list(models.keys()))

st.subheader("Введіть значення ознак")
user_input = {}
for column in X_train.columns:
    user_input[column] = st.number_input(f"{column}", value=0.0)

if st.button("Зробити прогноз"):
    input_data = np.array([list(user_input.values())]).astype(float)

    if selected_model == "Random Forest":
        transformed_data = input_data
    else:
        transformed_data = scaler.transform(input_data)

    model = loaded_models[selected_model]

    if selected_model == "Neural Network":
        prediction = model.predict(transformed_data)
        probability = prediction[0, 0]
        predicted_class = int(probability > 0.5)
    else:
        predicted_class = model.predict(transformed_data)[0]
        probability = model.predict_proba(transformed_data)[0][predicted_class]

    st.write(
        f"**Результат:** {'Залишається' if predicted_class == 0 else 'Йде'}")
    st.write(f"**Ймовірність:** {probability:.2%}")