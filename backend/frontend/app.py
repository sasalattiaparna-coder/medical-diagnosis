import gradio as gr
import requests

API_URL = "http://127.0.0.1:8000/predict"


def diagnose(image_path):

    with open(image_path, "rb") as img:

        files = {
            "file": img
        }

        response = requests.post(API_URL, files=files)

    result = response.json()

    prediction = result["prediction"]
    confidence = result["confidence"]

    return f"""
Prediction: {prediction}

Confidence: {confidence}%
"""


iface = gr.Interface(
    fn=diagnose,
    inputs=gr.Image(type="filepath", label="Upload Chest X-ray"),
    outputs=gr.Textbox(label="Diagnosis Result"),
    title="Medical AI Diagnosis System",
    description="Upload a chest X-ray image to detect COVID, NORMAL, or PNEUMONIA"
)

iface.launch()