import os
from pymongo import MongoClient

# -----------------------------------------
# Read MongoDB URI from environment variable
# -----------------------------------------
MONGODB_URI = os.getenv("MONGODB_URI")

if not MONGODB_URI:
    raise ValueError("MONGODB_URI environment variable is not set")

# -----------------------------------------
# Create MongoDB client
# -----------------------------------------
client = MongoClient(MONGODB_URI)

# -----------------------------------------
# Database & collections
# -----------------------------------------
db = client["chem_ai"]

users_col = db["users"]
history_col = db["history"]
