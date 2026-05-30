import streamlit as st
import pandas as pd
import joblib

model = joblib.load(
    "model/model.pkl"
)

features = joblib.load(
    "model/features.pkl"
)

st.set_page_config(
    page_title="Breast Cancer Prediction",
    layout="centered"
)

st.title(
    "Breast Cancer Prediction"
)

st.write(
    "Enter the tumor measurements below."
)

radius_mean = st.number_input(
    "Radius Mean",
    min_value=0.0,
    value=14.0
)

texture_mean = st.number_input(
    "Texture Mean",
    min_value=0.0,
    value=19.0
)

perimeter_mean = st.number_input(
    "Perimeter Mean",
    min_value=0.0,
    value=90.0
)

area_mean = st.number_input(
    "Area Mean",
    min_value=0.0,
    value=650.0
)

radius_worst = st.number_input(
    "Radius Worst",
    min_value=0.0,
    value=17.0
)

texture_worst = st.number_input(
    "Texture Worst",
    min_value=0.0,
    value=25.0
)

perimeter_worst = st.number_input(
    "Perimeter Worst",
    min_value=0.0,
    value=110.0
)

area_worst = st.number_input(
    "Area Worst",
    min_value=0.0,
    value=900.0
)

input_data = pd.DataFrame({
    "radius_mean": [radius_mean],
    "texture_mean": [texture_mean],
    "perimeter_mean": [perimeter_mean],
    "area_mean": [area_mean],
    "radius_worst": [radius_worst],
    "texture_worst": [texture_worst],
    "perimeter_worst": [perimeter_worst],
    "area_worst": [area_worst]
})

input_data = input_data[features]

if st.button("Predict"):

    prediction = model.predict(
        input_data
    )[0]

    probability = model.predict_proba(
        input_data
    )[0]

    if prediction == 1:
        st.error(
            "Prediction: Malignant"
        )
    else:
        st.success(
            "Prediction: Benign"
        )

    st.subheader(
        "Prediction Probability"
    )

    st.write(
        f"Benign: {probability[0] * 100:.2f}%"
    )

    st.write(
        f"Malignant: {probability[1] * 100:.2f}%"
    )