from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime

# -------------------------------
# App must start IMMEDIATELY
# -------------------------------
app = FastAPI(title="Chemistry Bot API")

# -------------------------------
# CORS (safe)
# -------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://chemibot.netlify.app/"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Healthcheck (Railway requirement)
# -------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# -------------------------------
# Root (optional but helpful)
# -------------------------------
@app.get("/")
def root():
    return {"status": "API running"}

# -------------------------------
# Schema (lightweight)
# -------------------------------
class Query(BaseModel):
    text: str

# -------------------------------
# Predict (IMPORT HEAVY CODE HERE)
# -------------------------------
@app.post("/predict")
def predict(q: Query):
    # import ONLY when endpoint is hit
    from model import generate_answer
    from database import history_col

    output = generate_answer(q.text)

    history_col.insert_one({
        "input": q.text,
        "output": output,
        "time": datetime.utcnow()
    })

    return {"output": output}

# -------------------------------
# History (DB import here only)
# -------------------------------
@app.get("/history")
def history():
    from database import history_col

    data = list(history_col.find().sort("time", -1).limit(10))
    return [
        {"input": d["input"], "output": d["output"]}
        for d in data
    ]
