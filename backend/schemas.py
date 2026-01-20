from pydantic import BaseModel

class Query(BaseModel):
    text: str

class Auth(BaseModel):
    email: str
    password: str
