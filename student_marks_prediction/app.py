from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "student_marks_model.joblib"
FEATURES = ["study_hours", "attendance", "previous_marks", "assignment_score"]

st.set_page_config(page_title="Student Marks Predictor", page_icon="📚")
st.title("📚 Student Marks Predictor")
st.write("Enter the student's details to estimate final marks.")
st.caption("Learning demo only. The prediction depends on the data used to train the model.")

if not MODEL_PATH.exists():
    st.warning("Model not found. First run `python train.py` in the project folder.")
    st.stop()

study_hours = st.number_input("Study hours per day", min_value=0.0, max_value=24.0, value=3.0, step=0.5)
attendance = st.slider("Attendance (%)", min_value=0, max_value=100, value=80)
previous_marks = st.slider("Previous marks (%)", min_value=0, max_value=100, value=65)
assignment_score = st.slider("Assignment score (%)", min_value=0, max_value=100, value=70)

if st.button("Predict marks", type="primary"):
    model = joblib.load(MODEL_PATH)
    sample = pd.DataFrame([{
        "study_hours": study_hours,
        "attendance": attendance,
        "previous_marks": previous_marks,
        "assignment_score": assignment_score,
    }], columns=FEATURES)
    estimate = float(model.predict(sample)[0])
    st.success(f"Estimated final marks: **{estimate:.1f} / 100**")
