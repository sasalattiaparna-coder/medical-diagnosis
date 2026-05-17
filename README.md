# Medical AI Diagnosis System

An AI-powered chest X-ray diagnosis system that predicts:

- COVID
- NORMAL
- PNEUMONIA

using FastAPI backend and Gradio frontend.

---

## Features

- Upload chest X-ray images
- Predict disease category
- Display confidence score
- FastAPI backend API
- Gradio interactive frontend
- Deep Learning model integration

---

## Technologies Used

- Python
- FastAPI
- Gradio
- PyTorch
- Requests
- Uvicorn

---

## Project Structure

```bash
medical-ai-diagnosis/
│
├── backend/
│   └── app.py
│
├── frontend/
│   └── app.py
│
├── models/
│   └── best_model.pth
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

Clone repository:

```bash
git clone https://github.com/your-username/medical-ai-diagnosis.git
```

Go to project folder:

```bash
cd medical-ai-diagnosis
```

Create virtual environment:

```bash
python -m venv venv
```

Activate virtual environment:

### Linux / Ubuntu

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run Backend

Go to backend folder:

```bash
cd backend
```

Run FastAPI server:

```bash
uvicorn app:app --reload
```

Backend runs on:

```bash
http://127.0.0.1:8000
```

---

## Run Frontend

Open another terminal.

Go to frontend folder:

```bash
cd frontend
```

Run frontend:

```bash
python app.py
```

Frontend runs on:

```bash
http://127.0.0.1:7860
```

---

## API Endpoint

### Predict Disease

```bash
POST /predict
```

Upload chest X-ray image and get prediction.

Example Response:

```json
{
    "prediction": "COVID",
    "confidence": 95.4
}
```

---

## Model Classes

- COVID
- NORMAL
- PNEUMONIA

---

## Future Improvements

- Real CNN model integration
- Grad-CAM visualization
- Model accuracy improvement
- Deployment using Docker
- Cloud hosting

---

## Author

Archana Sasalatti

```
