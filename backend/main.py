from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import os

# =========================================================
# APP
# =========================================================
app = FastAPI(title="Classroom Chemistry Bot")

# =========================================================
# CORS (NETLIFY + LOCALHOST SAFE)
# =========================================================
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https://.*\.netlify\.app|http://localhost:5173",
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

# =========================================================
# MODELS
# =========================================================
class Query(BaseModel):
    text: str

# =========================================================
# GLOBALS
# =========================================================
model = None
tokenizer = None
history_col = None

# =========================================================
# LOAD MODEL (LAZY)
# =========================================================
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

# =========================================================
# LOAD DATABASE
# =========================================================
def load_db():
    global history_col

    if history_col is None:
        from pymongo import MongoClient

        mongo_url = os.getenv("MONGODB_URL")
        if not mongo_url:
            raise RuntimeError("MONGODB_URL not set")

        client = MongoClient(mongo_url)
        db = client["chem_ai"]
        history_col = db["history"]

# =========================================================
# ROOT
# =========================================================
@app.get("/")
def root():
    return {"message": "Classroom Chemistry Bot API running"}

# =========================================================
# HEALTH
# =========================================================
@app.get("/health")
def health():
    return {"status": "ok"}

# =========================================================
# WARMUP (SERVER ONLY)
# =========================================================
@app.post("/warmup")
def warmup(request: Request):
    user_agent = request.headers.get("user-agent", "").lower()

    # Ignore browser calls
    if "mozilla" in user_agent:
        return {"status": "ignored"}

    load_model()
    load_db()
    return {"status": "warmed"}

# =========================================================
# PREDICT
# =========================================================
@app.post("/predict")
def predict(q: Query):
    load_model()
    load_db()

    import torch

    inputs = tokenizer(
        q.text,
        return_tensors="pt",
        truncation=True,
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=128,
            repetition_penalty=1.3,
            no_repeat_ngram_size=3,
        )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    history_col.insert_one({
        "input": q.text,
        "output": answer,
        "time": datetime.utcnow(),
    })

    return {"output": answer}

# =========================================================
# HISTORY
# =========================================================
@app.get("/history")
def history():
    load_db()

    data = list(
        history_col
        .find({}, {"_id": 0})
        .sort("time", -1)
        .limit(10)
    )

    return data
