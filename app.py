import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model.pkl")

st.set_page_config(page_title="Student Performance Predictor", layout="centered")

st.title("🎓 Student Performance Predictor")
st.write("Enter student details to predict final grade:")

# Inputs

gender = st.selectbox("Gender", ["Male", "Female"])

attendance = st.number_input("Attendance Rate (%)", min_value=0.0, max_value=100.0, value=75.0)

study_hours_week = st.number_input("Study Hours per Week", min_value=0.0, value=10.0)

previous_grade = st.number_input("Previous Grade", min_value=0.0, max_value=100.0, value=60.0)

extra = st.selectbox("Extracurricular Activities", ["Yes", "No"])

parent_support = st.selectbox("Parental Support", ["Low", "Medium", "High"])

study_hours = st.number_input("Daily Study Hours", min_value=0.0, value=2.0)

attendance_percent = st.number_input("Attendance (%) (Detailed)", min_value=0.0, max_value=100.0, value=75.0)

online_classes = st.number_input("Online Classes Taken", min_value=0, value=5)

# Predict
if st.button("Predict Final Grade"):

    input_df = pd.DataFrame({
        "gender": [gender],
        "attendancerate": [attendance],
        "studyhoursperweek": [study_hours_week],
        "previousgrade": [previous_grade],
        "extracurricularactivities": [extra],
        "parentalsupport": [parent_support],
        "study_hours": [study_hours],
        "attendance_(%)": [attendance_percent],
        "online_classes_taken": [online_classes]
    })

    pred = model.predict(input_df)

    st.success(f"📊 Predicted Final Grade: {pred[0]:.2f}")