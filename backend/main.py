from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import os
import traceback

app = FastAPI(title="Classroom Chemistry Bot")

# ================= CORS =================
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

# ================= MODELS =================
class Query(BaseModel):
    text: str

# ================= GLOBALS =================
_model = None
_tokenizer = None
_history_col = None

# ================= LOAD MODEL =================
def load_model():
    global _model, _tokenizer
    if _model is None:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        from peft import PeftModel
        import torch

        BASE_MODEL = "google/flan-t5-base"
        ADAPTER_PATH = "MyFinetunedModel"

        _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        base = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
        _model = PeftModel.from_pretrained(base, ADAPTER_PATH)
        _model.eval()

# ================= LOAD DB =================
def load_db():
    global _history_col
    if _history_col is None:
        from pymongo import MongoClient

        mongo_url = os.getenv("MONGODB_URL")
        if not mongo_url:
            raise RuntimeError("MONGODB_URL is not set")

        client = MongoClient(mongo_url)
        db = client["chem_ai"]
        _history_col = db["history"]

# ================= ROOT (FIXES 404 CONFUSION) =================
@app.get("/")
def root():
    return {
        "message": "Classroom Chemistry Bot API is running",
        "endpoints": ["/health", "/predict", "/history"]
    }

# ================= HEALTH =================
@app.get("/health")
def health():
    return {"status": "ok"}

# ================= PREDICT =================
@app.post("/predict")
def predict(q: Query):
    try:
        if not q.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")

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

    except HTTPException:
        raise
    except Exception as e:
        print("❌ Predict error:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal model error")

# ================= HISTORY =================
@app.get("/history")
def history():
    try:
        load_db()
        data = list(_history_col.find().sort("time", -1).limit(10))
        return [{"input": d["input"], "output": d["output"]} for d in data]
    except Exception:
        return []
