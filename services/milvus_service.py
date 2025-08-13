import os
import hashlib
from typing import List, Optional, Dict, Any
from pymilvus import (
    connections, Collection, CollectionSchema, FieldSchema, 
    DataType, utility, MilvusException
)
from .base import retry_on_failure, logging

class MilvusService:
    def __init__(self):
        self.uri = os.getenv("ZILLIZ_URI")
        self.token = os.getenv("ZILLIZ_TOKEN")
        self.collection_name = "trading_collection"
        self.collection = None
        self.logger = logging.getLogger(__name__)
        self._connection_established = False
        
    @retry_on_failure(max_retries=3, delay=2.0)
    def connect(self):
        """Establish connection to Milvus with retry logic"""
        if self._connection_established:
            return
            
        try:
            connections.connect(
                alias="default",
                uri=self.uri,
                token=self.token,
                timeout=30
            )
            self._connection_established = True
            self.logger.info("Connected to Milvus successfully")
        except Exception as e:
            self.logger.error(f"Failed to connect to Milvus: {e}")
            self._connection_established = False
            raise
    
    def ensure_connection(self):
        """Ensure we have a valid connection"""
        if not self._connection_established:
            self.connect()
    
    def get_or_create_collection(self, dim: int) -> Collection:
        """Get existing collection or create new one"""
        self.ensure_connection()
        
        try:
            if utility.has_collection(self.collection_name):
                collection = Collection(self.collection_name)
                # Validate schema
                if self._validate_schema(collection, dim):
                    collection.load()
                    self.collection = collection
                    self.logger.info(f"Loaded existing collection: {self.collection_name}")
                    return collection
                else:
                    # Drop and recreate if schema is invalid
                    self.logger.warning("Schema validation failed, recreating collection")
                    utility.drop_collection(self.collection_name)
            
            # Create new collection
            collection = self._create_collection(dim)
            self.collection = collection
            return collection
            
        except Exception as e:
            self.logger.error(f"Error with collection setup: {e}")
            raise
    
    def _validate_schema(self, collection: Collection, expected_dim: int) -> bool:
        """Validate collection schema matches expectations"""
        try:
            schema_fields = {field.name: field for field in collection.schema.fields}
            
            # Check required fields exist
            required_fields = {"id", "embedding", "text", "hash"}
            if not all(field in schema_fields for field in required_fields):
                return False
            
            # Check embedding dimension
            embedding_field = schema_fields.get("embedding")
            if embedding_field and hasattr(embedding_field, 'params'):
                actual_dim = embedding_field.params.get("dim", 0)
                if actual_dim != expected_dim:
                    return False
            
            return True
        except Exception:
            return False
    
    def _create_collection(self, dim: int) -> Collection:
        """Create new collection with proper schema"""
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2048),
            FieldSchema(name="hash", dtype=DataType.VARCHAR, max_length=64)  # For deduplication
        ]
        
        schema = CollectionSchema(fields, description="Trading data collection")
        collection = Collection(name=self.collection_name, schema=schema)
        
        # Create index
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "params": {"nlist": 128}
        }
        
        collection.create_index(field_name="embedding", index_params=index_params)
        collection.load()
        
        self.logger.info(f"Created new collection: {self.collection_name}")
        return collection
    
    def _generate_hash(self, text: str) -> str:
        """Generate hash for deduplication"""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    @retry_on_failure(max_retries=2)
    def check_existing_documents(self, texts: List[str]) -> List[str]:
        """Filter out existing documents based on hash"""
        if not self.collection:
            raise ValueError("Collection not initialized")
        
        new_texts = []
        
        try:
            for text in texts:
                text_hash = self._generate_hash(text)
                
                # Query by hash for exact match
                results = self.collection.query(
                    expr=f'hash == "{text_hash}"',
                    output_fields=["id"],
                    limit=1
                )
                
                if not results:
                    new_texts.append(text)
                    
        except Exception as e:
            self.logger.error(f"Error checking existing documents: {e}")
            # On error, return all texts to avoid data loss
            return texts
            
        self.logger.info(f"Found {len(new_texts)} new documents out of {len(texts)}")
        return new_texts
    
    @retry_on_failure(max_retries=2)
    def insert_documents(self, texts: List[str], embeddings: List[List[float]]):
        """Insert documents with embeddings"""
        if not self.collection:
            raise ValueError("Collection not initialized")
        
        if not texts or not embeddings:
            self.logger.warning("No documents to insert")
            return
            
        if len(texts) != len(embeddings):
            raise ValueError("Texts and embeddings length mismatch")
        
        try:
            # Generate hashes
            hashes = [self._generate_hash(text) for text in texts]
            
            entities = [embeddings, texts, hashes]
            
            insert_result = self.collection.insert(entities)
            self.collection.flush()
            
            self.logger.info(
                f"Inserted {insert_result.insert_count} documents. "
                f"Total entities: {self.collection.num_entities}"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to insert documents: {e}")
            raise
    
    @retry_on_failure(max_retries=2)
    def search(self, query_embedding: List[float], top_k: int = 3) -> List[str]:
        """Search for similar documents"""
        if not self.collection:
            raise ValueError("Collection not initialized")
        
        try:
            self.collection.load()  # Ensure collection is loaded
            
            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
            
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                output_fields=["text"]
            )
            
            if not results or not results[0]:
                self.logger.warning("No search results found")
                return []
            
            # Extract unique texts
            texts = []
            seen = set()
            
            for hit in results[0]:
                text = hit.entity.get("text", "")
                if text and text not in seen:
                    texts.append(text)
                    seen.add(text)
            
            self.logger.info(f"Retrieved {len(texts)} unique documents")
            return texts
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []