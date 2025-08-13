import os
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
)
from dotenv import load_dotenv
import numpy as np

load_dotenv()

ZILLIZ_URI = os.getenv("ZILLIZ_URI")
ZILLIZ_TOKEN = os.getenv("ZILLIZ_TOKEN")
COLLECTION_NAME = "trading_collection"


def connect_to_milvus():
    try:
        print("Connecting to Milvus...")
        connections.connect(uri=ZILLIZ_URI, token=ZILLIZ_TOKEN)
        print("Connected successfully:", connections.list_connections())
    except Exception as e:
        print(f"Failed to connect to Milvus: {str(e)}")
        raise


def create_collection(dim):
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1024),
    ]
    schema = CollectionSchema(fields, description="Gold trading documents")
    collection = Collection(name=COLLECTION_NAME, schema=schema)

    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128},
    }
    try:
        collection.create_index(field_name="embedding", index_params=index_params)
        print(f"Created collection '{COLLECTION_NAME}' with index")
    except Exception as e:
        print(f"Failed to create index: {str(e)}")
    return collection


def validate_collection_schema(dim):
    if utility.has_collection(COLLECTION_NAME):
        col = Collection(name=COLLECTION_NAME)
        schema = {f.name: f for f in col.schema.fields}
        if "embedding" not in schema or "text" not in schema:
            print(
                f"Collection '{COLLECTION_NAME}' missing required fields — recreating..."
            )
            utility.drop_collection(COLLECTION_NAME)
            return create_collection(dim)
        elif schema["embedding"].params.get("dim", dim) != dim:
            print(f"Collection '{COLLECTION_NAME}' has wrong dim — recreating...")
            utility.drop_collection(COLLECTION_NAME)
            return create_collection(dim)
        else:
            print(f"Collection '{COLLECTION_NAME}' schema is valid.")
            col.load()
            return col
    else:
        print(f"Collection '{COLLECTION_NAME}' does not exist — creating...")
        return create_collection(dim)


def get_or_create_collection(dim):
    try:
        return validate_collection_schema(dim)
    except Exception as e:
        print(f"Error accessing collection: {str(e)}")
        raise


def check_existing_documents(collection, text):
    safe = text.replace('"', r"\"")
    expr = f'text == "{safe}"'
    res = collection.query(expr=expr, output_fields=["text"])
    return len(res) > 0


def insert_documents(collection, texts, embeddings):
    if not texts or len(embeddings) == 0:
        print("No documents to insert")
        return

    if isinstance(embeddings, np.ndarray):
        embeddings = embeddings.tolist()

    print(f"Inserting {len(texts)} documents...")
    try:
        entities = [embeddings, texts]
        mr = collection.insert(entities)
        collection.flush()
        print(
            f"Inserted {mr.insert_count} documents, total entities: {collection.num_entities}"
        )
    except Exception as e:
        print(f"Failed to insert documents: {str(e)}")
        raise


def search(collection, query_embedding, top_k):
    print("Searching collection...")
    try:
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()
        collection.load()
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=top_k,
            output_fields=["text"],
        )
        hits = results[0]
        retrieved = list(dict.fromkeys([hit.entity.get("text") for hit in hits]))
        print(f"Retrieved {len(retrieved)} unique documents")
        return retrieved
    except Exception as e:
        print(f"Search failed: {str(e)}")
        return []
