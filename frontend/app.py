import gradio as gr
import requests


def diagnose(image):

    try:

        with open(image, "rb") as f:

            response = requests.post(
                "http://127.0.0.1:8000/predict",
                files={"file": f}
            )

        return response.text

    except Exception as e:

        return str(e)


iface = gr.Interface(
    fn=diagnose,
    inputs=gr.Image(type="filepath"),
    outputs="text"
)

iface.launch()