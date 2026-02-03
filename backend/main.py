from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import os

app = FastAPI(title="Classroom Chemistry Bot")

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://chemibot.netlify.app",
        "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- HEALTH ----------------
@app.get("/health")
def health():
    return {"status": "ok"}

# ---------------- REQUEST MODEL ----------------
class Query(BaseModel):
    text: str

# ---------------- GLOBALS ----------------
model = None
tokenizer = None
history_col = None

# ---------------- LOAD MODEL (LAZY) ----------------
def load_model():
    global model, tokenizer
    if model is None:
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        from peft import PeftModel

        BASE_MODEL = "google/flan-t5-base"
        ADAPTER_PATH = "MyFinetunedModel"

        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        model.eval()

# ---------------- LOAD DATABASE ----------------
def load_db():
    global history_col
    if history_col is None:
        from pymongo import MongoClient

        mongo_url = os.getenv("MONGODB_URL")
        if not mongo_url:
            raise RuntimeError("MONGODB_URL not set")

        client = MongoClient(
            mongo_url,
            tls=True,
            serverSelectionTimeoutMS=5000
        )
        db = client["chem_ai"]
        history_col = db["history"]

# ---------------- PREDICT ----------------
@app.post("/predict")
def predict(q: Query):
    try:
        load_model()
        load_db()

        import torch

        inputs = tokenizer(q.text, return_tensors="pt", truncation=True)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=128,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3
            )

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        history_col.insert_one({
            "input": q.text,
            "output": answer,
            "time": datetime.utcnow()
        })

        return {"output": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------- HISTORY ----------------
@app.get("/history")
def history():
    try:
        load_db()
        docs = list(history_col.find().sort("time", -1).limit(10))
        return [
            {"input": d["input"], "output": d["output"]}
            for d in docs
        ]
    except Exception:
        return []
