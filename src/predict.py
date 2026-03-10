from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path('models/model.pkl')


def main() -> None:
    model = joblib.load(MODEL_PATH)

    new_student = pd.DataFrame(
        [
            {
                'study_hours': 5,
                'attendance': 88,
                'sleep_hours': 7,
                'previous_grade': 82,
                'assignments_completed': 8,
            }
        ]
    )

    prediction = model.predict(new_student)[0]
    probability = model.predict_proba(new_student)[0][1]

    label = 'Strong performance' if prediction == 1 else 'Weak performance'
    print('Prediction:', label)
    print('Probability:', round(float(probability), 2))


if __name__ == '__main__':
    main()
