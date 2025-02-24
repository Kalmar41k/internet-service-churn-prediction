import streamlit as st
import pickle
import tensorflow as tf
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

X_train = pd.read_csv("data/X_train.csv")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

models = {
    "Logistic Regression": "models/logistic_regression_model.pkl",
    "Random Forest": "models/random_forest_model.pkl",
    "SVM": "models/best_svm_model.pkl",
    "Neural Network": "models/dnn_model.h5"
}

loaded_models = {}
for name, path in models.items():
    if name == "Neural Network":
        loaded_models[name] = tf.keras.models.load_model(path)
    elif name == "SVM":
        loaded_models[name] = joblib.load(path)
    else:
        with open(path, "rb") as f:
            loaded_models[name] = pickle.load(f)

st.title("Прогнозування відтоку клієнтів")
selected_model = st.selectbox("Оберіть модель", list(models.keys()))

st.subheader("Введіть значення ознак")
user_input = {}
for column in X_train.columns:
    user_input[column] = st.number_input(f"{column}", value=0.0)

if st.button("Зробити прогноз"):
    input_data = pd.DataFrame([user_input], columns=X_train.columns)

    if selected_model == "Random Forest":
        transformed_data = input_data.values
    else:
        transformed_data = scaler.transform(input_data)

    model = loaded_models[selected_model]

    if selected_model == "Neural Network":
        prediction = model.predict(transformed_data)[0, 0]
        predicted_class = int(prediction > 0.5)
        probability = prediction if predicted_class == 1 else (1 - prediction)
    else:
        predicted_class = model.predict(transformed_data)[0]
        probability = model.predict_proba(transformed_data)[0][predicted_class]

    st.write(
        f"**Результат:** {'Залишається' if predicted_class == 0 else 'Йде'}")
    st.write(f"**Ймовірність:** {probability:.2%}")