from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import os

# -------------------------------------------------
# IMPORTANT: create app FIRST (no heavy imports yet)
# -------------------------------------------------
app = FastAPI()

# -------------------------------------------------
# CORS (Netlify frontend + Railway backend)
# -------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://chemibot.netlify.app",
        "https://classroomchemistrybot-production.up.railway.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# FAST endpoints (Railway health checks)
# -------------------------------------------------
@app.get("/")
def root():
    return {
        "status": "Chemistry Bot API running",
        "message": "Backend is alive"
    }

@app.get("/health")
def health():
    return {"status": "ok"}

# -------------------------------------------------
# Lazy imports (ONLY after startup is complete)
# -------------------------------------------------
from auth import router
from schemas import Query
from database import history_col
from model import generate_answer

app.include_router(router)

# -------------------------------------------------
# Prediction endpoint (ML runs ONLY here)
# -------------------------------------------------
@app.post("/predict")
def predict(q: Query):
    try:
        output = generate_answer(q.text)
    except Exception:
        # Railway-safe error instead of timeout
        raise HTTPException(
            status_code=503,
            detail="Model is loading or busy. Please try again in a moment."
        )

    history_col.insert_one({
        "input": q.text,
        "output": output,
        "time": datetime.utcnow()
    })

    return {"output": output}

# -------------------------------------------------
# History endpoint
# -------------------------------------------------
@app.get("/history")
def history():
    data = list(history_col.find().sort("time", -1).limit(10))
    return [
        {"input": d["input"], "output": d["output"]}
        for d in data
    ]
