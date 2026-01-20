from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017")
db = client["chem_ai"]

users_col = db["users"]
history_col = db["history"]
