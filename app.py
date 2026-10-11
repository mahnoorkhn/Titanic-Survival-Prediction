import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Titanic Survival Prediction", page_icon="🚢")
st.title("🚢 Titanic Passenger Survival Prediction")
st.write("Enter passenger details to estimate whether the passenger survived.")

DATA_PATH = Path(__file__).parent / "sample-data.csv"
MODEL_PATH = Path(__file__).parent / "svc_trained_model.pkl"

required_columns = ["PClass", "Gender", "Sibling", "Embarked", "Survived"]

if not DATA_PATH.exists():
    st.error("sample-data.csv is missing. Add it to the same GitHub repository as app.py.")
    st.stop()

if not MODEL_PATH.exists():
    st.error("svc_trained_model.pkl is missing. Run the notebook through the model-saving cell, then upload this file to GitHub.")
    st.stop()

data = pd.read_csv(DATA_PATH)
missing_columns = [column for column in required_columns if column not in data.columns]
if missing_columns:
    st.error("The dataset is missing these columns: " + ", ".join(missing_columns))
    st.write("Columns found:", list(data.columns))
    st.stop()

data = data[required_columns].dropna().copy()

pclass_encoder = LabelEncoder().fit(data["PClass"].astype(str))
gender_encoder = LabelEncoder().fit(data["Gender"].astype(str))
sibling_encoder = LabelEncoder().fit(data["Sibling"].astype(str))
embarked_encoder = LabelEncoder().fit(data["Embarked"].astype(str))
survived_encoder = LabelEncoder().fit(data["Survived"].astype(str))

with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)

st.subheader("Passenger details")
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Passenger class", list(pclass_encoder.classes_))
    gender = st.selectbox("Gender", list(gender_encoder.classes_))

with col2:
    sibling = st.selectbox("Siblings / spouses aboard", list(sibling_encoder.classes_))
    embarked = st.selectbox("Port of embarkation", list(embarked_encoder.classes_))

if st.button("Predict survival", type="primary"):
    try:
        input_data = pd.DataFrame({
            "PClass": [pclass_encoder.transform([str(pclass)])[0]],
            "Gender": [gender_encoder.transform([str(gender)])[0]],
            "Sibling": [sibling_encoder.transform([str(sibling)])[0]],
            "Embarked": [embarked_encoder.transform([str(embarked)])[0]],
        })
        prediction = int(model.predict(input_data)[0])
        result = survived_encoder.inverse_transform([str(prediction)])[0]
        normalized_result = str(result).strip().lower()

        if normalized_result in {"1", "yes", "survived", "true"}:
            st.success("Prediction: The passenger is predicted to have survived.")
        elif normalized_result in {"0", "no", "not survived", "died", "false"}:
            st.warning("Prediction: The passenger is predicted not to have survived.")
        else:
            st.info(f"Predicted target value: {result}")

        st.caption("This is a machine-learning estimate, not a certainty.")
    except Exception as error:
        st.error(f"Prediction failed: {error}")
        st.write("Check that the dataset categories match the categories used when training the model.")
