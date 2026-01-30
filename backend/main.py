from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from auth import router
from schemas import Query
from model import generate_answer
from database import history_col
from datetime import datetime

app = FastAPI()

# ================= CORS =================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Netlify + future domains
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= ROUTERS =================
app.include_router(router)

# ================= HEALTH CHECK (REQUIRED FOR RAILWAY) =================
@app.get("/health")
def health():
    return {"status": "ok"}

# ================= PREDICT =================
@app.post("/predict")
def predict(q: Query):
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
