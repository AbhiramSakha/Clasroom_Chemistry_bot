import os
from pymongo import MongoClient

MONGO_URI = os.getenv("MONGO_URI")

client = MongoClient(MONGO_URI)
db = client["chem_ai"]

users_col = db["users"]
history_col = db["history"]
