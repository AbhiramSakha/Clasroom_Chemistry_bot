from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from auth import router
from schemas import Query
from model import generate_answer
from database import history_col

app = FastAPI()

# -----------------------------
# CORS (Netlify only)
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://chemibot.netlify.app",
        "https://clasroomchemistrybot-production.up.railway.app"
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# -----------------------------
# Root (NEVER touches ML)
# -----------------------------
@app.get("/")
def root():
    return {
        "status": "Chemistry Bot API running",
        "endpoints": ["/health", "/predict", "/history"]
    }

# -----------------------------
# Health check (instant)
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# -----------------------------
# Predict (ML ONLY here)
# -----------------------------
@app.post("/predict")
def predict(q: Query):
    try:
        output = generate_answer(q.text)
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail="Model is busy or loading. Please retry."
        )

    history_col.insert_one({
        "input": q.text,
        "output": output,
        "time": datetime.utcnow()
    })

    return {"output": output}

# -----------------------------
# History
# -----------------------------
@app.get("/history")
def history():
    data = list(history_col.find().sort("time", -1).limit(10))
    return [
        {"input": d["input"], "output": d["output"]}
        for d in data
    ]
