import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import time
import warnings
warnings.filterwarnings('ignore')

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        font-family: 'Arial', sans-serif;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
    }
    .stProgress .st-bo {
        background-color: #2196F3;
    }
    h1, h2 {
        color: #2c3e50;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .sidebar .sidebar-content {
        background-color: #34495e;
        color: white;
    }
    .prediction-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
    }
    .metric-text {
        color: #e74c3c;
        font-size: 24px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Title with emoji
st.write("# 🏋️ Personal Fitness Tracker", anchor=None)
st.write("Track your calorie burn with style! Input your parameters and get personalized predictions.")

# Sidebar styling
st.sidebar.markdown("<h2 style='color: #ffffff;'>User Input Parameters</h2>", unsafe_allow_html=True)

def user_input_features():
    age = st.sidebar.slider("Age", 10, 100, 30, help="Your age in years")
    bmi = st.sidebar.slider("BMI", 15, 40, 20, help="Body Mass Index")
    duration = st.sidebar.slider("Duration (min)", 0, 35, 15, help="Exercise duration")
    heart_rate = st.sidebar.slider("Heart Rate", 60, 130, 80, help="Beats per minute")
    body_temp = st.sidebar.slider("Body Temp (°C)", 36, 42, 38, help="Temperature in Celsius")
    gender_button = st.sidebar.radio("Gender", ("Male", "Female"), help="Select your gender")

    gender = 1 if gender_button == "Male" else 0

    data_model = {
        "Age": age,
        "BMI": bmi,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
        "Gender_male": gender
    }
    return pd.DataFrame(data_model, index=[0])

df = user_input_features()

# Display parameters
st.markdown("---")
st.header("📊 Your Parameters")
with st.spinner('Loading your inputs...'):
    progress_bar = st.progress(0)
    for i in range(100):
        progress_bar.progress(i + 1)
        time.sleep(0.005)
    st.dataframe(df.style.set_properties(**{
        'background-color': '#ffffff',
        'border-color': '#ddd',
        'border-radius': '5px',
        'padding': '5px'
    }))

# Load and preprocess data
calories = pd.read_csv("calories.csv")
exercise = pd.read_csv("exercise.csv")

exercise_df = exercise.merge(calories, on="User_ID")
exercise_df.drop(columns="User_ID", inplace=True)

exercise_train_data, exercise_test_data = train_test_split(exercise_df, test_size=0.2, random_state=1)

for data in [exercise_train_data, exercise_test_data]:
    data["BMI"] = data["Weight"] / ((data["Height"] / 100) ** 2)
    data["BMI"] = round(data["BMI"], 2)

exercise_train_data = exercise_train_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
exercise_test_data = exercise_test_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
exercise_train_data = pd.get_dummies(exercise_train_data, columns=['Gender'], drop_first=True)
exercise_test_data = pd.get_dummies(exercise_test_data, columns=['Gender'], drop_first=True)

X_train = exercise_train_data.drop("Calories", axis=1)
y_train = exercise_train_data["Calories"]
X_test = exercise_test_data.drop("Calories", axis=1)
y_test = exercise_test_data["Calories"]

# Train model
random_reg = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=6)
random_reg.fit(X_train, y_train)

df = df.reindex(columns=X_train.columns, fill_value=0)
prediction = random_reg.predict(df)

# Prediction display
st.markdown("---")
st.header("🔥 Prediction")
with st.spinner('Calculating your calorie burn...'):
    progress_bar = st.progress(0)
    for i in range(100):
        progress_bar.progress(i + 1)
        time.sleep(0.005)
    st.markdown(f"""
        <div class='prediction-box'>
            <span class='metric-text'>{round(prediction[0], 2)}</span> kilocalories burned
        </div>
    """, unsafe_allow_html=True)

# Similar results
st.markdown("---")
st.header("👥 Similar Results")
with st.spinner('Finding similar profiles...'):
    progress_bar = st.progress(0)
    for i in range(100):
        progress_bar.progress(i + 1)
        time.sleep(0.005)
    
    calorie_range = [prediction[0] - 10, prediction[0] + 10]
    similar_data = exercise_df[(exercise_df["Calories"] >= calorie_range[0]) & 
                             (exercise_df["Calories"] <= calorie_range[1])]
    if not similar_data.empty:
        st.dataframe(similar_data.sample(5).style.set_properties(**{
            'background-color': '#ffffff',
            'border-color': '#ddd',
            'border-radius': '5px'
        }))
    else:
        st.write("No similar results found")

# General stats
st.markdown("---")
st.header("📈 Your Stats")
cols = st.columns(4)
boolean_age = (exercise_df["Age"] < df["Age"].values[0]).mean() * 100
boolean_duration = (exercise_df["Duration"] < df["Duration"].values[0]).mean() * 100
boolean_body_temp = (exercise_df["Body_Temp"] < df["Body_Temp"].values[0]).mean() * 100
boolean_heart_rate = (exercise_df["Heart_Rate"] < df["Heart_Rate"].values[0]).mean() * 100

with cols[0]:
    st.markdown(f"<p style='color: #3498db;'>Age Rank<br><b>{round(boolean_age, 1)}%</b></p>", unsafe_allow_html=True)
with cols[1]:
    st.markdown(f"<p style='color: #3498db;'>Duration<br><b>{round(boolean_duration, 1)}%</b></p>", unsafe_allow_html=True)
with cols[2]:
    st.markdown(f"<p style='color: #3498db;'>Heart Rate<br><b>{round(boolean_heart_rate, 1)}%</b></p>", unsafe_allow_html=True)
with cols[3]:
    st.markdown(f"<p style='color: #3498db;'>Body Temp<br><b>{round(boolean_body_temp, 1)}%</b></p>", unsafe_allow_html=True)