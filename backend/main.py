from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import os

app = FastAPI(title="Classroom Chemistry Bot")

# ------------------ CORS (FIXED) ------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://chemibot.netlify.app",
        "http://localhost:5173",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ HEALTH ------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# ------------------ REQUEST MODEL ------------------
class Query(BaseModel):
    text: str

# ------------------ GLOBALS ------------------
_model = None
_tokenizer = None
_history_col = None

# ------------------ LOAD MODEL (SAFE) ------------------
def load_model():
    global _model, _tokenizer
    if _model is not None:
        return

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        from peft import PeftModel

        BASE_MODEL = "google/flan-t5-base"
        ADAPTER_PATH = "MyFinetunedModel"

        _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
        _model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        _model.eval()

    except Exception as e:
        print("❌ Model load failed:", e)
        raise HTTPException(status_code=500, detail="Model loading failed")

# ------------------ LOAD DB (TLS FIX) ------------------
def load_db():
    global _history_col
    if _history_col is not None:
        return

    try:
        from pymongo import MongoClient

        mongo_url = os.getenv("MONGODB_URL")
        if not mongo_url:
            print("⚠️ MongoDB disabled (no URL)")
            return

        client = MongoClient(
            mongo_url,
            tls=True,
            tlsAllowInvalidCertificates=True,
            serverSelectionTimeoutMS=5000
        )

        db = client["chem_ai"]
        _history_col = db["history"]

    except Exception as e:
        print("❌ MongoDB connection failed:", e)
        _history_col = None

# ------------------ PREDICT ------------------
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

    # Save history ONLY if DB is available
    if _history_col:
        try:
            _history_col.insert_one({
                "input": q.text,
                "output": answer,
                "time": datetime.utcnow()
            })
        except Exception as e:
            print("⚠️ History save failed:", e)

    return {"output": answer}

# ------------------ HISTORY ------------------
@app.get("/history")
def history():
    load_db()

    if not _history_col:
        return []

    try:
        data = list(_history_col.find().sort("time", -1).limit(10))
        return [{"input": d["input"], "output": d["output"]} for d in data]
    except Exception as e:
        print("⚠️ History fetch failed:", e)
        return []
