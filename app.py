import gradio as gr
import joblib
import numpy as np
from datetime import datetime

# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = joblib.load("models/best_aqi_model.pkl")

# ============================================================
# AQI CATEGORIZATION
# ============================================================

def categorize_aqi(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"

# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_aqi(pm25, pm10, no2, co, o3, year, month, day):

    try:
        features = np.array([[pm25, pm10, no2, co, o3, year, month, day]])
        prediction = model.predict(features)[0]
        category = categorize_aqi(prediction)

        if prediction > 200:
            warning = "⚠ ALERT: Immediate Action Required"
        elif prediction > 100:
            warning = "⚠ Moderate Risk – Sensitive Groups Should Be Careful"
        else:
            warning = "Air Quality Acceptable"

        result = (
            f"Predicted AQI: {round(prediction,2)}\n"
            f"Category: {category}\n"
            f"{warning}"
        )

        return result

    except Exception as e:
        return f"Error: {str(e)}"

# ============================================================
# CLASSICAL SIMPLE UI
# ============================================================

interface = gr.Interface(
    fn=predict_aqi,
    inputs=[
        gr.Number(label="PM2.5"),
        gr.Number(label="PM10"),
        gr.Number(label="NO2"),
        gr.Number(label="CO"),
        gr.Number(label="O3"),
        gr.Number(label="Year"),
        gr.Number(label="Month"),
        gr.Number(label="Day"),
    ],
    outputs=gr.Textbox(label="AQI Prediction Result"),
    title="AI for Earth - AQI Prediction System",
    description="Enter pollution values and date to predict AQI and risk level."
)

# ============================================================
# RUN APP
# ============================================================

if __name__ == "__main__":
    interface.launch()