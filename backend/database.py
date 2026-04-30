from pymongo import MongoClient

client = MongoClient("mongodb+srv://sakhabhiram1234_db_user:VuEf7lDZeIIw7iab@cluster0.ezt3bs1.mongodb.net/?appName=Cluster0")
db = client["chem_ai"]

users_col = db["users"]
history_col = db["history"]
