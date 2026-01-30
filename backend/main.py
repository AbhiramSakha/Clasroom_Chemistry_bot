from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import os

from model import generate_answer
from database import history_col
from auth import router as auth_router

# =========================
# APP
# =========================
app = FastAPI(title="Classroom Chemistry Bot API")

# =========================
# CORS (Netlify frontend)
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://chemibot.netlify.app",
        "https://clasroomchemistrybot.netlify.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# ROUTERS
# =========================
app.include_router(auth_router)

# =========================
# SCHEMAS
# =========================
class Query(BaseModel):
    text: str

# =========================
# HEALTH CHECK (Railway)
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}

# =========================
# PREDICT (POST ONLY)
# =========================
@app.post("/predict")
def predict(q: Query):
    output = generate_answer(q.text)

    history_col.insert_one({
        "input": q.text,
        "output": output,
        "time": datetime.utcnow()
    })

    return {"output": output}

# =========================
# HISTORY
# =========================
@app.get("/history")
def history():
    data = list(history_col.find().sort("time", -1).limit(10))
    return [
        {"input": d["input"], "output": d["output"]}
        for d in data
    ]
