import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

from auth import router
from schemas import Query
from model import generate_answer
from database import history_col

app = FastAPI()

# ✅ CORS – allow Netlify frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://chemibot.netlify.app",
        "https://clasroomchemistrybot-production.up.railway.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

@app.get("/")
def root():
    return {"status": "Chemistry AI Backend Running"}

@app.post("/predict")
def predict(q: Query):
    output = generate_answer(q.text)
    history_col.insert_one({
        "input": q.text,
        "output": output,
        "time": datetime.utcnow()
    })
    return {"output": output}

@app.get("/history")
def history():
    data = list(history_col.find().sort("time", -1).limit(10))
    return [{"input": d["input"], "output": d["output"]} for d in data]
