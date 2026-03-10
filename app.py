from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path('models/model.pkl')

st.set_page_config(page_title='AI Student Performance Predictor', page_icon='🎓')
st.title('AI Student Performance Predictor')
st.write('Enter student details to predict academic performance.')

if not MODEL_PATH.exists():
    st.warning('Model not found. Please run `python src/train.py` first.')
    st.stop()

model = joblib.load(MODEL_PATH)

study_hours = st.slider('Study Hours per Day', 0, 12, 5)
attendance = st.slider('Attendance (%)', 0, 100, 80)
sleep_hours = st.slider('Sleep Hours', 0, 12, 7)
previous_grade = st.slider('Previous Grade', 0, 100, 75)
assignments_completed = st.slider('Assignments Completed', 0, 10, 7)

if st.button('Predict'):
    input_data = pd.DataFrame(
        [
            {
                'study_hours': study_hours,
                'attendance': attendance,
                'sleep_hours': sleep_hours,
                'previous_grade': previous_grade,
                'assignments_completed': assignments_completed,
            }
        ]
    )

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f'High chance of strong performance ({probability:.2%})')
    else:
        st.error(f'Risk of weak performance ({1 - probability:.2%})')
