from fastapi import FastAPI, UploadFile, File
import random

app = FastAPI()


@app.get("/")
def home():
    return {"message": "Backend Working"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    classes = ["COVID", "NORMAL", "PNEUMONIA"]

    prediction = random.choice(classes)

    confidence = round(random.uniform(90, 99), 2)

    return {
        "prediction": prediction,
        "confidence": confidence
    }