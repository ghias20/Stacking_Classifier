import streamlit as st
import pickle
import pandas as pd

model = pickle.load(
    open("models/stacking_classifier.pkl", "rb")
)

st.title("Iris Flower Classification using Stacking")

sepal_length = st.number_input("Sepal Length")
sepal_width = st.number_input("Sepal Width")
petal_length = st.number_input("Petal Length")
petal_width = st.number_input("Petal Width")

if st.button("Predict"):

    data = pd.DataFrame({
        "sepal length (cm)": [sepal_length],
        "sepal width (cm)": [sepal_width],
        "petal length (cm)": [petal_length],
        "petal width (cm)": [petal_width]
    })

    prediction = model.predict(data)

    classes = {
        0: "Setosa",
        1: "Versicolor",
        2: "Virginica"
    }

    st.success(
        f"Predicted Flower : {classes[prediction[0]]}"
    )