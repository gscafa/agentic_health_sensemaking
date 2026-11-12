from pymongo import MongoClient, ASCENDING
import pandas as pd
import os
from pathlib import Path

processed_user_data_dir = Path(".").resolve() / "data" / "processed" / "users"

client = MongoClient("mongodb://localhost:27017/")


try:
  client.admin.command("ping")
  print("Connected to MongoDB")
except Exception as e:
  print(f"Connection error: {e}")
  exit(1)

db = client["ahs_ces_data"]

daily_db_collection = db["daily_data"]
hourly_db_collection = db["hourly_data"]
ema_db_collection = db["ema_data"]

daily_db_collection.drop()
hourly_db_collection.drop()
ema_db_collection.drop()

def insert_csv_to_mongo(csv_path: Path, collection):
  if not csv_path.exists():
    return

  temp_df = pd.read_csv(csv_path)
  temp_dict = temp_df.to_dict(orient="records")
  if temp_dict:
    cleaned_dict = [
        {k: (None if pd.isna(v) else v) for k, v in doc.items()}
        for doc in temp_dict
    ]
    
    collection.insert_many(cleaned_dict)

for current_user_dir in processed_user_data_dir.iterdir():
  if(current_user_dir.is_dir()):
    insert_csv_to_mongo(current_user_dir / "daily.csv", daily_db_collection)
    insert_csv_to_mongo(current_user_dir / "hourly.csv", hourly_db_collection)
    insert_csv_to_mongo(current_user_dir / "ema.csv", ema_db_collection)

daily_db_collection.create_index([("uid", ASCENDING), ("day", ASCENDING)])
hourly_db_collection.create_index([("uid", ASCENDING), ("day", ASCENDING), ("hour", ASCENDING)])
ema_db_collection.create_index([("uid", ASCENDING), ("day", ASCENDING)])
