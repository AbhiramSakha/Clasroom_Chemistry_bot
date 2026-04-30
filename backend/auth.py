from fastapi import APIRouter, HTTPException
from schemas import Auth
from database import users_col
from werkzeug.security import generate_password_hash, check_password_hash

router = APIRouter()

@router.post("/signup")
def signup(data: Auth):
    if users_col.find_one({"email": data.email}):
        raise HTTPException(status_code=400, detail="User exists")

    users_col.insert_one({
        "email": data.email,
        "password": generate_password_hash(data.password)
    })
    return {"msg": "Signup success"}

@router.post("/login")
def login(data: Auth):
    user = users_col.find_one({"email": data.email})
    if not user or not check_password_hash(user["password"], data.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return {"msg": "Login success", "email": data.email}
