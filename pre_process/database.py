import pandas as pd
import json
from mongo_config import DATABASE_NAME, MONGO_URI
from pymongo import MongoClient



def insert_data_csv(collection_name, csv_path, id_unique = 'id'):
    client = MongoClient(MONGO_URI)
    # Select the database
    db = client[DATABASE_NAME]

    df = pd.read_csv(csv_path)
    if id_unique not in df.columns:
        df[id_unique] = range(1, len(df) + 1)
    # print(df[id_unique])
    if collection_name not in db.list_collection_names():
        db.create_collection(collection_name)
    df.drop(columns=[df.columns[0]], inplace=True, errors='ignore')
    data = df.to_dict(orient='records')
    collection = db[collection_name]
    for record in data:
        # Assuming there is a unique identifier in your data, replace 'unique_field_name' with the actual field name
        unique_field_value = record[id_unique]
        
        # Use update_one with upsert=True to insert or update the record
        collection.update_one(
            filter={id_unique: unique_field_value},
            update={'$set': record},
            upsert=True
        )
    client.close()
    print(f"Data {csv_path} has been successfully inserted or updated in MongoDB Atlas.")

def insert_data_json(collection_name, csv_path, id_unique = 'id'):
    client = MongoClient(MONGO_URI)
    # Select the database
    db = client[DATABASE_NAME]

    with open(csv_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    
    if collection_name not in db.list_collection_names():
        db.create_collection(collection_name)
    collection = db[collection_name]

    for item in data:
        collection.update_one(
            {id_unique: item[id_unique]}, 
            {'$set': item},
            upsert=True
        )

    # Đóng kết nối MongoDB
    client.close()

def append_data_csv(collection_name, csv_path, id_unique = 'id'):
    client = MongoClient(MONGO_URI)
    # Select the database
    db = client[DATABASE_NAME]

    df = pd.read_csv(csv_path)
    if id_unique not in df.columns:
        df[id_unique] = range(1, len(df) + 1)
    if collection_name not in db.list_collection_names():
        db.create_collection(collection_name)
    df.drop(columns=[df.columns[0]], inplace=True, errors='ignore')
    data = df.to_dict(orient='records')
    collection = db[collection_name]
    collection.insert_many(data, ordered=False)
    client.close()
    print(f"Data {csv_path} has been successfully appended in MongoDB.")

def append_data_json(collection_name, csv_path, id_unique = 'id'):
    client = MongoClient(MONGO_URI)
    # Select the database
    db = client[DATABASE_NAME]
    with open(csv_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    if collection_name not in db.list_collection_names():
        db.create_collection(collection_name)
    collection = db[collection_name]
    collection.insert_many(data, ordered=False)
    client.close()
    print(f"Data {csv_path} has been successfully inserted or updated in MongoDB Atlas.")

# ls_processed_data = os.listdir('./proccessed')
# for path in ls_processed_data:
#     if '.json' in path:
#         name, _ = os.path.splitext(path)
#         append_data_json(name, f'proccessed/{path}', id_unique = 'id')
#
# for path in ls_processed_data:
#     if '.csv' in path:
#         name, _ = os.path.splitext(path)
#         append_data_csv(name, f'proccessed/{path}', id_unique = 'unique_id')


