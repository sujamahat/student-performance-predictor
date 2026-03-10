# AI Student Performance Predictor

A beginner-friendly machine learning project that predicts whether a student is likely to show strong academic performance based on daily habits and past results.

## Problem Statement

Students and educators often want an early signal of academic risk. This project trains a binary classification model to predict if a student is likely to perform strongly (`1`) or weakly (`0`) using practical, explainable inputs.

## Project Structure

```text
student-performance-predictor/
├── data/
│   └── student_data.csv
├── models/
│   └── model.pkl
├── notebooks/
│   └── exploration.ipynb
├── src/
│   ├── train.py
│   └── predict.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Dataset Features

The dataset is stored in `data/student_data.csv`.

- `study_hours`: hours studied per day
- `attendance`: attendance percentage
- `sleep_hours`: sleep duration per day
- `previous_grade`: previous overall grade
- `assignments_completed`: assignment completion score/count proxy
- `performance`: target label (`1 = strong`, `0 = weak`)

## Model Used

- **Algorithm:** Logistic Regression (`scikit-learn`)
- **Split:** 80/20 train-test split with stratification
- **Output:** model artifact saved to `models/model.pkl`

## Setup

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Train the Model

```bash
python src/train.py
```

Example output:

- Accuracy score
- Classification report
- Confirmation of model save path

## Run a CLI Prediction

```bash
python src/predict.py
```

Expected style of output:

- `Prediction: Strong performance`
- `Probability: 0.xx`

## Run the Streamlit App

```bash
streamlit run app.py
```

In the app, move the sliders and click **Predict** to get:

- **High chance of strong performance** with probability, or
- **Risk of weak performance** with risk probability

## Screenshot

_Add your app screenshot here after running Streamlit._

## Future Improvements

- Add more rows and richer real-world features
- Try Random Forest, XGBoost, or calibrated models
- Add model explainability (feature importance / SHAP)
- Provide personalized study recommendations
- Deploy with Streamlit Community Cloud or Render
