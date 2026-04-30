from pydantic import BaseModel
from typing import Optional

class Query(BaseModel):
    text: str
    language: Optional[str] = "en"   # e.g. "en", "te", "hi", "ta"

class Auth(BaseModel):
    email: str
    password: str

class Query(BaseModel):
    text: str

class Auth(BaseModel):
    email: str
    password: str
