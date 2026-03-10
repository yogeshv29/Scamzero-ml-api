from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AlbertForSequenceClassification, AlbertTokenizer
import torch

app = FastAPI()

MODEL_PATH = "./saved_model"

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

import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("api:app", host="0.0.0.0", port=port)
