from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime

from model import generate_answer
from database import history_col
from auth import router as auth_router

app = FastAPI(title="Classroom Chemistry Bot API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # safe for now
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)


class Query(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "ok"}

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
