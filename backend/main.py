import os
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from auth import router
from schemas import Query
from model import generate_answer
from database import history_col

# ================= APP INIT =================
app = FastAPI(title="Chemistry AI Backend")

# ================= CORS =================
# Allow Netlify frontend + local dev (optional)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://chemibot.netlify.app",
        "https://chemibot.netlify.app/",
        "https://clasroomchemistrybot-production.up.railway.app",   # optional (local dev)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= ROUTERS =================
app.include_router(router)

# ================= ROOT (IMPORTANT) =================
@app.get("/")
def root():
    return {"status": "Chemistry AI Backend is running"}

# ================= HEALTH CHECK (REQUIRED BY RAILWAY) =================
@app.get("/health")
def health():
    return {"status": "ok"}

# ================= MODEL API =================
@app.post("/predict")
def predict(q: Query):
    output = generate_answer(q.text)

    history_col.insert_one({
        "input": q.text,
        "output": output,
        "time": datetime.utcnow()
    })

    return {"output": output}

# ================= HISTORY API =================
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
