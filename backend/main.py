from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime

from database import history_col
from auth import router as auth_router

app = FastAPI(title="Classroom Chemistry Bot API")

# ================= CORS =================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)

# ================= SCHEMA =================
class Query(BaseModel):
    text: str

# ================= HEALTH (MUST BE FIRST & LIGHT) =================
@app.get("/health")
def health():
    return {"status": "ok"}

# ================= PREDICT =================
@app.post("/predict")
def predict(q: Query):
    # 🚨 IMPORT MODEL ONLY WHEN REQUEST COMES
    from model import generate_answer

    output = generate_answer(q.text)

    history_col.insert_one({
        "input": q.text,
        "output": output,
        "time": datetime.utcnow()
    })

    return {"output": output}

# ================= HISTORY =================
@app.get("/history")
def history():
    data = list(history_col.find().sort("time", -1).limit(10))
    return [{"input": d["input"], "output": d["output"]} for d in data]
