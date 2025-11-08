import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import gradio as gr

# =======================
# 1️⃣ Dataset
# =======================
data = {
    'Study_Hours': [1, 2, 3, 4, 5, 6, 7, 8],
    'Attendance': [60, 65, 70, 75, 80, 85, 90, 95],
    'Score': [35, 40, 50, 55, 65, 70, 80, 90]
}
df = pd.DataFrame(data)

X = df[['Study_Hours', 'Attendance']]
y = df['Score']

# =======================
# 2️⃣ Train Model
# =======================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# =======================
# 3️⃣ Evaluation Metrics
# =======================
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# =======================
# 4️⃣ Prediction Function
# =======================
def predict_score(study_hours, attendance):
    data = pd.DataFrame({'Study_Hours': [study_hours], 'Attendance': [attendance]})
    prediction = model.predict(data)[0]

    # Create scatter plot
    plt.figure(figsize=(5, 3))
    plt.scatter(y_test, y_pred, color='blue', label="Predicted vs Actual")
    plt.xlabel("Actual Scores")
    plt.ylabel("Predicted Scores")
    plt.title("Model Performance")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plot.png")
    plt.close()

    return (
        f"🎯 Predicted Score: {round(prediction, 2)}",
        f"📊 MAE: {mae:.2f} | MSE: {mse:.2f} | R²: {r2:.2f}",
        "plot.png"
    )

# =======================
# 5️⃣ Gradio Interface
# =======================
interface = gr.Interface(
    fn=predict_score,
    inputs=[
        gr.Slider(0, 10, label="Study Hours"),
        gr.Slider(50, 100, label="Attendance (%)")
    ],
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Model Metrics"),
        gr.Image(label="Performance Plot")
    ],
    title="📘 Student Score Predictor",
    description=(
        "A machine learning app that predicts student exam scores based on "
        "study hours and attendance. Built and deployed by **Altaf Raja**.<br><br>"
        "🔗 [GitHub](https://github.com/Altafraja21) | "
        "💼 [LinkedIn](https://linkedin.com/in/altaf-raja21) | "
        "🌐 [Portfolio](https://altaf-raja-ul.onrender.com)"
    ),
    theme="gradio/soft"
)

interface.launch()
