from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AlbertForSequenceClassification, AlbertTokenizer
import torch
import os

app = FastAPI()

# safer path for deployment environments
MODEL_PATH = os.path.join(os.path.dirname(__file__), "saved_model")

# load tokenizer and model
tokenizer = AlbertTokenizer.from_pretrained(MODEL_PATH)
model = AlbertForSequenceClassification.from_pretrained(MODEL_PATH)

model.eval()

class JobInput(BaseModel):
    description: str


@app.get("/")
def read_root():
    return {
        "message": "ScamZero Prediction API is running!",
        "endpoints": {
            "predict": "/predict (POST)",
            "health": "/health (GET)"
        }
    }


@app.get("/health")
def health_check():
    return {"status": "healthy", "model": "albert-base-v2"}


@app.post("/predict")
def predict(data: JobInput):

    text = data.description

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = model(**inputs)

        probs = torch.softmax(outputs.logits, dim=-1)
        prediction = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][prediction].item()

    return {
        "prediction": prediction,
        "label": "Scam" if prediction == 1 else "Safe",
        "confidence": confidence
    }
