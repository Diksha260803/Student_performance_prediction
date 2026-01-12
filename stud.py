import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("student_data.csv")

# Features and Target
X = data[['Hours_Studied', 'Attendance', 'Previous_Score']]
y = data['Final_Score']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# User Input Prediction
print("\n--- Student Performance Prediction ---")
hours = float(input("Enter hours studied: "))
attendance = float(input("Enter attendance percentage: "))
previous_score = float(input("Enter previous exam score: "))

prediction = model.predict([[hours, attendance, previous_score]])
print("Predicted Final Score:", round(prediction[0], 2))
