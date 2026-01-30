import torch
torch.set_num_threads(1)
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from auth import router
from schemas import Query
from model import generate_answer
from database import history_col


# -------------------------------------------------
# App initialization
# -------------------------------------------------
app = FastAPI()


# -------------------------------------------------
# CORS configuration (NO localhost)
# -------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://chemibot.netlify.app",
        "https://clasroomchemistrybot-production.up.railway.app"
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------
# Routers
# -------------------------------------------------
app.include_router(router)


# -------------------------------------------------
# Root endpoint (fixes Railway error)
# -------------------------------------------------
@app.get("/")
def root():
    return {
        "message": "Chemistry Bot Backend is running 🚀",
        "frontend": "https://chemibot.netlify.app",
        "endpoints": ["/predict", "/history", "/health"]
    }


# -------------------------------------------------
# Health check endpoint
# -------------------------------------------------
@app.get("/health")
def health():
    return {"status": "Backend running ✅"}


# -------------------------------------------------
# Prediction endpoint
# -------------------------------------------------
@app.post("/predict")
def predict(q: Query):
    output = generate_answer(q.text)

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
        {
            "input": d.get("input"),
            "output": d.get("output")
        }
        for d in data
    ]
