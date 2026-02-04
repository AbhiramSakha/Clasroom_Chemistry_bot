import os
from pymongo import MongoClient

MONGODB_URL = os.getenv("MONGODB_URL")

client = MongoClient(MONGODB_URL)
db = client["chem_ai"]

users_col = db["users"]
history_col = db["history"]
