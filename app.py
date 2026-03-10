from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from transformers import AlbertTokenizer, AlbertForSequenceClassification
import re
import os

app = FastAPI(title="Job Scam Detection ML API", version="1.0.0")

# --- Performance Optimization ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "saved_model")

# Load model and tokenizer
model_loaded = False
try:
    if os.path.exists(MODEL_PATH) and os.listdir(MODEL_PATH):
        print(f"Loading cached ALBERT model from {MODEL_PATH}...")
        tokenizer = AlbertTokenizer.from_pretrained(MODEL_PATH)
        model = AlbertForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(device)
        model.eval()
        model_loaded = True
        print("Model loaded successfully!")
    else:
        print(f"No trained model found at {MODEL_PATH}.")
except Exception as e:
    print(f"Failed to load model: {e}")

class JobPosting(BaseModel):
    title: str = ""
    description: str = ""
    requirements: str = ""
    benefits: str = ""
    company_profile: str = ""

def clean_text(text: str) -> str:
    """Matches the exact cleaning logic used in training."""
    text = re.sub(r"<[^>]+>", " ", text)   # strip HTML
    text = re.sub(r"[^\w\s]", " ", text)   # remove special chars
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict_job_scam(title, description, requirements, benefits, company_profile):
    # Combine text fields (order should match training)
    job_text = f"{title} {description} {requirements} {benefits} {company_profile}"
    job_text = clean_text(job_text)
    
    if not model_loaded:
        # Fallback to a very safe score if no model exists
        return 0.5, "Suspicious (Model not loaded)"

    # Tokenize
    inputs = tokenizer(job_text, return_tensors="pt", max_length=256, padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Run ALBERT inference
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        # Index 1 is Fake Job
        prob = probs[0][1].item()
            
    # Output classification
    if prob < 0.30:
        classification = "Safe"
    elif prob <= 0.70:
        classification = "Suspicious"
    else:
        classification = "Fake"
        
    return prob, classification

@app.post("/detect-scam")
async def detect_scam(job: JobPosting):
    try:
        scam_probability, classification = predict_job_scam(
            job.title, 
            job.description, 
            job.requirements, 
            job.benefits, 
            job.company_profile
        )
        return {
            "scam_probability": round(scam_probability, 4),
            "classification": classification
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model_loaded}
