from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

DATA_PATH = Path('data/student_data.csv')
MODEL_PATH = Path('models/model.pkl')
FEATURES = [
    'study_hours',
    'attendance',
    'sleep_hours',
    'previous_grade',
    'assignments_completed',
]
TARGET = 'performance'


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, zero_division=0))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f'Model saved to {MODEL_PATH}')


if __name__ == '__main__':
    main()
