import streamlit as st
import pandas as pd
import joblib


model = joblib.load('model.pkl')
features = joblib.load('features.pkl')

st.set_page_config(page_title='Breast Cancer Prediction', layout='wide')

st.title('Breast Cancer Prediction using Decision Tree')
st.write('Enter the tumor details below')

col1, col2, col3 = st.columns(3)

inputs = {}

with col1:
    inputs['radius_mean'] = st.slider('Radius Mean', 5.0, 30.0, 15.0)
    inputs['texture_mean'] = st.slider('Texture Mean', 5.0, 40.0, 20.0)
    inputs['perimeter_mean'] = st.slider('Perimeter Mean', 40.0, 200.0, 90.0)
    inputs['area_mean'] = st.slider('Area Mean', 100.0, 2500.0, 700.0)
    inputs['smoothness_mean'] = st.slider('Smoothness Mean', 0.05, 0.2, 0.1)
    inputs['compactness_mean'] = st.slider('Compactness Mean', 0.0, 0.5, 0.1)
    inputs['concavity_mean'] = st.slider('Concavity Mean', 0.0, 0.5, 0.1)
    inputs['concave points_mean'] = st.slider('Concave Points Mean', 0.0, 0.3, 0.05)
    inputs['symmetry_mean'] = st.slider('Symmetry Mean', 0.0, 0.5, 0.2)
    inputs['fractal_dimension_mean'] = st.slider('Fractal Dimension Mean', 0.0, 0.2, 0.06)

with col2:
    inputs['radius_se'] = st.slider('Radius SE', 0.0, 3.0, 0.5)
    inputs['texture_se'] = st.slider('Texture SE', 0.0, 5.0, 1.0)
    inputs['perimeter_se'] = st.slider('Perimeter SE', 0.0, 25.0, 3.0)
    inputs['area_se'] = st.slider('Area SE', 0.0, 600.0, 40.0)
    inputs['smoothness_se'] = st.slider('Smoothness SE', 0.0, 0.05, 0.01)
    inputs['compactness_se'] = st.slider('Compactness SE', 0.0, 0.2, 0.03)
    inputs['concavity_se'] = st.slider('Concavity SE', 0.0, 0.2, 0.03)
    inputs['concave points_se'] = st.slider('Concave Points SE', 0.0, 0.1, 0.01)
    inputs['symmetry_se'] = st.slider('Symmetry SE', 0.0, 0.1, 0.02)
    inputs['fractal_dimension_se'] = st.slider('Fractal Dimension SE', 0.0, 0.05, 0.005)

with col3:
    inputs['radius_worst'] = st.slider('Radius Worst', 5.0, 40.0, 20.0)
    inputs['texture_worst'] = st.slider('Texture Worst', 5.0, 50.0, 25.0)
    inputs['perimeter_worst'] = st.slider('Perimeter Worst', 40.0, 300.0, 120.0)
    inputs['area_worst'] = st.slider('Area Worst', 100.0, 5000.0, 1000.0)
    inputs['smoothness_worst'] = st.slider('Smoothness Worst', 0.05, 0.3, 0.15)
    inputs['compactness_worst'] = st.slider('Compactness Worst', 0.0, 1.5, 0.3)
    inputs['concavity_worst'] = st.slider('Concavity Worst', 0.0, 1.5, 0.3)
    inputs['concave points_worst'] = st.slider('Concave Points Worst', 0.0, 0.5, 0.15)
    inputs['symmetry_worst'] = st.slider('Symmetry Worst', 0.0, 1.0, 0.3)
    inputs['fractal_dimension_worst'] = st.slider('Fractal Dimension Worst', 0.0, 0.3, 0.08)

input_df = pd.DataFrame([inputs])
input_df = input_df[features]

if st.button('Predict'):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    if prediction == 1:
        st.error('Prediction: Malignant')
    else:
        st.success('Prediction: Benign')

    st.subheader('Prediction Probability')
    st.write(f'Benign: {probability[0] * 100:.2f}%')
    st.write(f'Malignant: {probability[1] * 100:.2f}%')