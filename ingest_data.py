import os
import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# Configuration
DB_URL = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "synthetic_documents")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

def ingest_to_postgres():
    print("Connecting to PostgreSQL...")
    engine = create_engine(DB_URL)
    
    # Load CSVs
    crm_df = pd.read_csv("source_data/crm_customers.csv")
    events_df = pd.read_csv("source_data/clickstream_events.csv")
    
    # Convert timestamps
    crm_df['created_at'] = pd.to_datetime(crm_df['created_at'])
    events_df['event_timestamp'] = pd.to_datetime(events_df['event_timestamp'])
    
    # Ingest to Postgres
    print("Ingesting CRM data...")
    crm_df.to_sql("customers", engine, if_exists="replace", index=False)
    
    print("Ingesting events data...")
    events_df.to_sql("events", engine, if_exists="replace", index=False)
    
    print("PostgreSQL ingestion complete.")
    return crm_df

def calculate_interests(cust_events):
    if cust_events.empty:
        return "No specific behavioral interests calculated."
    
    weights = {'purchase': 3, 'add_to_cart': 2, 'view': 1}
    product_scores = {}
    color_scores = {}
    
    for _, e in cust_events.iterrows():
        weight = weights.get(e['event_type'], 1)
        
        # Score products
        prod = e['product']
        product_scores[prod] = product_scores.get(prod, 0) + weight
        
        # Score colors
        color = e['color']
        color_scores[color] = color_scores.get(color, 0) + weight
        
    # Get top 2 products and colors
    top_products = sorted(product_scores.items(), key=lambda x: x[1], reverse=True)[:2]
    top_colors = sorted(color_scores.items(), key=lambda x: x[1], reverse=True)[:2]
    
    interests = f"Primary interests: {', '.join([p[0] for p in top_products])}. "
    interests += f"Preferred colors: {', '.join([c[0] for c in top_colors])}."
    return interests

def ingest_to_qdrant(crm_df, events_df):
    print("Connecting to Qdrant...")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    # Recreate collection
    print(f"Creating collection: {COLLECTION_NAME}...")
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
    
    print("Generating embeddings and uploading to Qdrant...")
    points = []
    for _, row in crm_df.iterrows():
        customer_id = row['customer_id']
        
        # Get customer events and calculate interests
        cust_events = events_df[events_df['customer_id'] == customer_id]
        calculated_interests = calculate_interests(cust_events)
        
        # Luxury Tagging based on spend
        luxury_tag = ""
        if row['total_spent'] > 800:
            luxury_tag = " This customer likes luxury items."
        
        # Construct a semantic description of the customer
        description = (
            f"Customer {row['first_name']} {row['last_name']} ({row['email']}) "
            f"from {row['country']}, age {row['age']}, favorite color {row['favorite_color']}. "
            f"Total spent: {row['total_spent']}. {calculated_interests}{luxury_tag}"
        )
        
        vector = model.encode(description).tolist()
        
        payload = {
            "customer_id": int(customer_id),
            "text": description,
            "metadata": {
                **row.to_dict(),
                "country": row['country'], # Duplicate for filtering
                "calculated_interests": calculated_interests,
                "likes_luxury": bool(luxury_tag)
            }
        }
        
        points.append(PointStruct(
            id=int(customer_id),
            vector=vector,
            payload=payload
        ))
        
        if len(points) >= 100:
            client.upsert(collection_name=COLLECTION_NAME, points=points)
            points = []
            
    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        
    print("Qdrant ingestion complete.")

if __name__ == "__main__":
    engine = create_engine(DB_URL)
    crm_df = pd.read_csv("source_data/crm_customers.csv")
    events_df = pd.read_csv("source_data/clickstream_events.csv")
    
    # Ingest to Postgres
    crm_df.to_sql("customers", engine, if_exists="replace", index=False)
    events_df.to_sql("events", engine, if_exists="replace", index=False)
    
    ingest_to_qdrant(crm_df, events_df)
