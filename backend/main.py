from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import os

# ================== APP ==================
app = FastAPI(title="Classroom Chemistry Bot API")

# ================== CORS ==================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://chemibot.netlify.app",   # Netlify frontend
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================== ROOT (IMPORTANT) ==================
@app.get("/")
def root():
    return {
        "message": "Classroom Chemistry Bot API is running 🚀",
        "endpoints": {
            "health": "/health",
            "predict": "POST /predict",
            "history": "GET /history",
            "docs": "/docs"
        }
    }

# ================== HEALTH ==================
@app.get("/health")
def health():
    return {"status": "ok"}

# ================== REQUEST SCHEMA ==================
class Query(BaseModel):
    text: str

# ================== GLOBAL LAZY OBJECTS ==================
_model = None
_tokenizer = None
_history_col = None

# ================== LOAD MODEL (LAZY) ==================
def load_model():
    global _model, _tokenizer

    if _model is None:
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        from peft import PeftModel

        BASE_MODEL = "google/flan-t5-base"
        ADAPTER_PATH = "MyFinetunedModel"

        _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
        _model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        _model.eval()

# ================== LOAD DATABASE (LAZY) ==================
def load_db():
    global _history_col

    if _history_col is None:
        from pymongo import MongoClient

        mongo_url = os.environ.get("MONGO_URI")
        if not mongo_url:
            raise RuntimeError("MONGO_URI environment variable not set")

        client = MongoClient(mongo_url)
        db = client["chem_ai"]
        _history_col = db["history"]

# ================== PREDICT ==================
@app.post("/predict")
def predict(q: Query):
    load_model()
    load_db()

    import torch

    inputs = _tokenizer(q.text, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_length=128,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3
        )

    answer = _tokenizer.decode(outputs[0], skip_special_tokens=True)

    _history_col.insert_one({
        "input": q.text,
        "output": answer,
        "time": datetime.utcnow()
    })

    return {"output": answer}

# ================== HISTORY ==================
@app.get("/history")
def history():
    load_db()

    data = list(
        _history_col.find()
        .sort("time", -1)
        .limit(10)
    )

    return [
        {"input": d["input"], "output": d["output"]}
        for d in data
    ]
