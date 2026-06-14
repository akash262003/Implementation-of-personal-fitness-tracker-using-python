# Personal Fitness Tracker — Python & Streamlit

## Overview

A machine learning-powered web application built with Python and Streamlit that predicts calories burned during exercise based on personal health parameters. Users can input their age, BMI, workout duration, heart rate, and body temperature to receive an instant calorie burn prediction.

- **Language:** Python
- **Framework:** Streamlit
- **ML Models:** Linear Regression, Random Forest Regressor
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Dataset:** Exercise and calorie data (15,000+ records)

---

## Features

- Real-time calorie burn prediction based on user inputs
- Choice between two ML models: Linear Regression and Random Forest
- Interactive sidebar with sliders for Age, BMI, Duration, Heart Rate, Body Temperature
- Model performance metrics displayed: R² Score and RMSE
- Data visualisations: Calories Distribution histogram and Feature Correlation heatmap
- Downloadable prediction results as CSV

---

## Machine Learning

| Model | Description |
|---|---|
| Linear Regression | Baseline model — fast, interpretable |
| Random Forest Regressor | Ensemble model — higher accuracy (`n_estimators=100`, `max_depth=6`) |

**Target variable:** Calories burned (continuous — regression problem)

**Input features:**
- Age
- BMI (calculated from Weight and Height)
- Exercise Duration (minutes)
- Heart Rate (bpm)
- Body Temperature (°C)
- Gender (Male/Female — label encoded)

---

## Model Performance

| Metric | Value |
|---|---|
| R² Score | ~0.99 (Random Forest) |
| RMSE | Low — accurate prediction within ~2 kcal |

---

## Project Structure

```
Implementation-of-personal-fitness-tracker-using-python/
├── fitness_tracker.py     ← Main Streamlit app (Linear Regression + Random Forest)
├── app.py                 ← Alternative app version (Random Forest only)
├── calories.csv           ← Calorie dataset
├── exercise.csv           ← Exercise dataset
├── requirements.txt       ← Python dependencies
└── README.md
```

---

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/akash262003/Implementation-of-personal-fitness-tracker-using-python.git
cd Implementation-of-personal-fitness-tracker-using-python
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run fitness_tracker.py
```

4. Open your browser at `http://localhost:8501`

---

## Dataset

| File | Records | Description |
|---|---|---|
| `exercise.csv` | 15,000 | User exercise sessions (Age, Weight, Height, Duration, Heart Rate, Body Temp) |
| `calories.csv` | 15,000 | Corresponding calories burned per session |

Both datasets are merged on `User_ID` during preprocessing.

---

## How It Works

```
User Input (Sidebar)
        ↓
Data Preprocessing (BMI calculation, one-hot encoding)
        ↓
Model Training (Train/Test split 80/20)
        ↓
Prediction → Calories Burned
        ↓
Performance Metrics + Visualisations
        ↓
Optional CSV Download
```

---

## Skills Demonstrated

- Streamlit web application development
- Machine learning with scikit-learn (regression)
- Data preprocessing and feature engineering
- Model evaluation (R² Score, RMSE)
- Data visualisation with Matplotlib and Seaborn
- Object-oriented programming and input validation

---

## Author

**Akash Saravanan**
- GitHub: [github.com/akash262003](https://github.com/akash262003)
- LinkedIn: [linkedin.com/in/akash26623](https://linkedin.com/in/akash26623)
- Email: akash26623@gmail.com
