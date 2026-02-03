from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import os

# ------------------ APP ------------------
app = FastAPI(title="Classroom Chemistry Bot")

# ------------------ CORS (MUST BE FIRST) ------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://chemibot.netlify.app",
        "https://clasroomchemistrybot-production.up.railway.app",
        "http://localhost:5173"
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
    if _model is None:
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
            print("❌ Model load error:", e)
            raise HTTPException(status_code=500, detail="Model loading failed")

# ------------------ LOAD DB (SAFE) ------------------
def load_db():
    global _history_col
    if _history_col is None:
        try:
            from pymongo import MongoClient

            mongo_url = os.environ.get(
                "MONGODB_URL",
                "mongodb+srv://sakhabhiram1234_db_user:VuEf7lDZeIIw7iab@cluster0.ezt3bs1.mongodb.net/?retryWrites=true&w=majority"
            )

            client = MongoClient(
                mongo_url,
                serverSelectionTimeoutMS=5000,
                tls=True,
                tlsAllowInvalidCertificates=True
            )

            db = client["chem_ai"]
            _history_col = db["history"]

        except Exception as e:
            print("❌ MongoDB error:", e)
            _history_col = None  # prevent crash

# ------------------ PREDICT ------------------
@app.post("/predict")
def predict(q: Query):
    load_model()
    load_db()

    if _model is None or _tokenizer is None:
        raise HTTPException(status_code=500, detail="Model not available")

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

    if _history_col:
        _history_col.insert_one({
            "input": q.text,
            "output": answer,
            "time": datetime.utcnow()
        })

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
        print("❌ History error:", e)
        return []
