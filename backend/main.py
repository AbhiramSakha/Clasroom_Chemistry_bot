from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import os

# ================= APP =================
app = FastAPI(title="Classroom Chemistry Bot API")

# ================= CORS =================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://chemibot.netlify.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= ROOT =================
@app.get("/")
def root():
    return {
        "message": "Classroom Chemistry Bot API is running",
        "endpoints": {
            "health": "/health",
            "warmup": "POST /warmup",
            "predict": "POST /predict",
            "history": "GET /history"
        }
    }

# ================= HEALTH =================
@app.get("/health")
def health():
    return {"status": "ok"}

# ================= REQUEST SCHEMA =================
class Query(BaseModel):
    text: str

# ================= GLOBAL LAZY OBJECTS =================
_model = None
_tokenizer = None
_history_col = None

# ================= LOAD MODEL (LAZY) =================
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

# ================= LOAD DATABASE =================
def load_db():
    global _history_col

    if _history_col is None:
        from pymongo import MongoClient

        mongo_url = os.environ.get("MONGODB_URL")
        if not mongo_url:
            raise RuntimeError("MONGODB_URL not set")

        client = MongoClient(
            mongo_url,
            tls=True,
            tlsAllowInvalidCertificates=True,
            serverSelectionTimeoutMS=30000
        )

        client.admin.command("ping")
        db = client["chem_ai"]
        _history_col = db["history"]

# ================= WARMUP =================
@app.post("/warmup")
def warmup():
    try:
        load_model()
        return {"status": "model loaded"}
    except Exception as e:
        return {"status": "failed", "error": str(e)}

# ================= PREDICT =================
@app.post("/predict")
def predict(q: Query):
    try:
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

    except Exception as e:
        raise HTTPException(status_code=500, detail="Prediction failed")

# ================= HISTORY =================
@app.get("/history")
def history():
    try:
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

    except Exception:
        raise HTTPException(status_code=500, detail="History fetch failed")
