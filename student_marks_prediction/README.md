# Student Marks Prediction

A beginner-friendly regression project that predicts a student's final marks from study hours, attendance, previous marks, and assignment score. It uses Python, pandas, and scikit-learn, with an optional Streamlit interface.

> `data/student_scores.csv` is a small **synthetic demonstration dataset** so the project runs immediately. Replace it with a suitable Kaggle dataset before drawing real-world conclusions.

## Requirements

- Python 3.10+

## Setup

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Train and evaluate

```bash
python train.py
```

The script saves the trained model to `models/student_marks_model.joblib` and evaluation metrics to `reports/metrics.json`. It reports MAE, RMSE, and R² on a held-out test set.

## Run the web app

```bash
streamlit run app.py
```

Enter the student's details and click **Predict marks**.

## Use a Kaggle dataset

Download a student performance/scores CSV from Kaggle and replace `data/student_scores.csv`. Ensure it has these numeric columns (or rename/map its columns first):

- `study_hours`: hours studied
- `attendance`: attendance percentage, 0–100
- `previous_marks`: prior marks, 0–100
- `assignment_score`: assignment marks, 0–100
- `final_marks`: target marks, 0–100

Then rerun `python train.py`. A dataset with different column names needs a small mapping/cleaning step in `train.py`; the current version intentionally keeps the first project simple.

## Project files

- `train.py` — loads data, splits train/test, trains a Random Forest regressor, evaluates and saves it
- `app.py` — simple prediction form
- `data/student_scores.csv` — synthetic demo data
- `reports/metrics.json` — created by training
- `models/` — saved model created by training

## Limitations

This is a learning demo, not a reliable way to assess real students. The sample data is synthetic, and model quality depends on representative data, sound features, and careful evaluation. Avoid using sensitive personal data without permission.
