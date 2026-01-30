import time
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any

from pymongo import MongoClient, ASCENDING, DESCENDING
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

# --- Configuration ---
MONGO_URI = "mongodb://localhost:27017/"
MONGO_DB_NAME = "agent_memory_v1"
QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION = "memories"
VECTOR_SIZE = 384 # Using 'all-MiniLM-L6-v2' for demo (normally 768 or 1536)

class MemoryStoreInit:
    def __init__(self):
        # 1. Connect to MongoDB
        self.mongo_client = MongoClient(MONGO_URI)
        self.db = self.mongo_client[MONGO_DB_NAME]
        
        # 2. Connect to Qdrant
        self.qdrant_client = QdrantClient(
            url=QDRANT_URL
        )
        
        # 3. Load Embedding Model
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

    def initialize_schema(self):
        """Sets up MongoDB indexes and Qdrant collection."""
        print("Initializing Schema...")
        
        # --- MongoDB Indexes ---
        # Messages: Fast session reconstruction
        self.db.messages.create_index([("session_id", ASCENDING), ("created_at", ASCENDING)])
        
        # Memories: Retrieval by user, type, and importance
        self.db.memory_nodes.create_index([
            ("user_id", ASCENDING), 
            ("type", ASCENDING), 
            ("importance", DESCENDING)
        ])
        # Memories: Cleanup/Pruning
        self.db.memory_nodes.create_index([("last_used_at", ASCENDING)])

        # --- Qdrant Collection ---
        if not self.qdrant_client.collection_exists(QDRANT_COLLECTION):
            self.qdrant_client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=models.VectorParams(
                    size=VECTOR_SIZE,
                    distance=models.Distance.COSINE
                )
            )
            # Create payload indexes for filtering
            self.qdrant_client.create_payload_index(
                collection_name=QDRANT_COLLECTION,
                field_name="user_id",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
            self.qdrant_client.create_payload_index(
                collection_name=QDRANT_COLLECTION,
                field_name="session_id",
                field_schema=models.PayloadSchemaType.KEYWORD
            )
        print("Schema Initialization Complete.")

    def get_embedding(self, text: str) -> List[float]:
        return self.encoder.encode(text).tolist()

class MemoryService:
    def __init__(self, store: MemoryStoreInit):
        self.mongo = store.db
        self.qdrant = store.qdrant_client
        self.embed_fn = store.get_embedding

    # --- 1. Ingestion (Raw Messages) ---
    def add_message(self, user_id: str, session_id: str, role: str, content: str):
        """Log a raw chat message to MongoDB."""
        msg_doc = {
            "_id": str(uuid.uuid4()),
            "session_id": session_id,
            "user_id": user_id,
            "role": role,
            "content": content,
            "created_at": datetime.utcnow()
        }
        self.mongo.messages.insert_one(msg_doc)
        return msg_doc["_id"]

    # --- 2. Memory Promotion (Summary -> Mongo & Qdrant) ---
    def promote_memory(self, user_id: str, session_id: str, 
                       summary: str, importance: float, 
                       m_type: str = "episodic", raw_refs: List[str] = None):
        """
        Creates a 'Memory Node'. 
        Always writes to Mongo.
        Only writes to Qdrant if importance > threshold (e.g., 0.5).
        """
        memory_id = str(uuid.uuid4())
        timestamp = datetime.utcnow()
        
        # 1. Write to MongoDB (Source of Truth)
        mongo_node = {
            "_id": memory_id,
            "user_id": user_id,
            "session_id": session_id,
            "type": m_type, # episodic, semantic, preference
            "summary": summary,
            "raw_refs": raw_refs or [],
            "importance": importance,
            "last_used_at": timestamp,
            "created_at": timestamp
        }
        self.mongo.memory_nodes.insert_one(mongo_node)

        # 2. Write to Qdrant (Semantic Cache) if important enough
        # Threshold can be adjusted. Here we index everything > 0.3 for demo.
        if importance > 0.3:
            vector = self.embed_fn(summary)
            
            payload = {
                "memory_id": memory_id,
                "user_id": user_id,
                "session_id": session_id,
                "type": m_type,
                "importance": importance,
                "created_at": timestamp.isoformat(),
                "summary_preview": summary[:100] # store snippet for debugging
            }
            
            self.qdrant.upsert(
                collection_name=QDRANT_COLLECTION,
                points=[
                    models.PointStruct(
                        id=str(uuid.uuid4()), # Qdrant Point ID
                        vector=vector,
                        payload=payload
                    )
                ]
            )
            print(f"Memory promoted to Qdrant: {summary[:30]}...")

    # --- 3. Retrieval (The Hybrid Algorithm) ---
    def retrieve_context(self, user_id: str, session_id: str, query: str):
        """
        Performs the dual-stage retrieval using the Unified Query API.
        """
        query_vector = self.embed_fn(query)
        
        # A. Session-Local Recall (Filter by current session_id)
        session_response = self.qdrant.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_vector,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id)),
                    models.FieldCondition(key="session_id", match=models.MatchValue(value=session_id))
                ]
            ),
            limit=3
        )
        session_hits = session_response.points

        # B. Cross-Session / Global Recall (Exclude current session, high importance)
        global_response = self.qdrant.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_vector,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id)),
                    models.FieldCondition(key="importance", range=models.Range(gte=0.7)) 
                ],
                must_not=[
                    models.FieldCondition(key="session_id", match=models.MatchValue(value=session_id))
                ]
            ),
            limit=3
        )
        global_hits = global_response.points

        # C. Fuse & Fetch Details
        all_hits = session_hits + global_hits
        unique_memory_ids = set()
        final_memories = []

        for hit in all_hits:
            mem_id = hit.payload["memory_id"]
            if mem_id not in unique_memory_ids:
                unique_memory_ids.add(mem_id)
                
                # Fetch full details from MongoDB (Source of Truth)
                full_node = self.mongo.memory_nodes.find_one({"_id": mem_id})
                if full_node:
                    final_memories.append(full_node)
                    self._touch_memory(mem_id)

        return final_memories

    def _touch_memory(self, memory_id: str):
        """Updates the last_used_at timestamp in MongoDB."""
        self.mongo.memory_nodes.update_one(
            {"_id": memory_id},
            {"$set": {"last_used_at": datetime.utcnow()}}
        )

    # --- 4. Short-Term Buffer ---
    def get_recent_messages(self, session_id: str, limit: int = 10):
        """Fetches the raw chat history window."""
        cursor = self.mongo.messages.find(
            {"session_id": session_id}
        ).sort("created_at", DESCENDING).limit(limit)
        
        return list(cursor)[::-1]
    
    # --- 5. Maintenance / Cleanup ---
    def clear_memories(self, user_id: str = None):
        """
        Clears memories from both databases.
        
        Args:
            user_id (str): If provided, deletes only that user's memories.
                           If None, deletes EVERYTHING (Nuclear option).
        """
        if user_id:
            print(f"Deleting memories for User: {user_id}...")
            
            # 1. MongoDB: Delete specific user's nodes
            # Note: We usually keep raw 'messages' for audit, but delete 'memory_nodes'
            mongo_result = self.mongo.memory_nodes.delete_many({"user_id": user_id})
            
            # 2. Qdrant: Delete points by Filter
            # We use FilterSelector to delete by criteria
            qdrant_result = self.qdrant.delete(
                collection_name=QDRANT_COLLECTION,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="user_id", 
                                match=models.MatchValue(value=user_id)
                            )
                        ]
                    )
                )
            )
            print(f"Deleted {mongo_result.deleted_count} docs from Mongo.")
            print(f"Deleted points from Qdrant for user {user_id}.")

        else:
            print("WARNING: Deleting ALL memories (Nuclear Option)...")
            
            # 1. MongoDB: Truncate collection
            self.mongo.memory_nodes.delete_many({})
            
            # 2. Qdrant: Delete everything (Empty filter matches all)
            self.qdrant.delete(
                collection_name=QDRANT_COLLECTION,
                points_selector=models.FilterSelector(
                    filter=models.Filter()
                )
            )
            print("All memories wiped.")
    
# --- Setup ---
store = MemoryStoreInit()
store.initialize_schema()
memory_service = MemoryService(store)

USER_ID = "user_123"
SESSION_ID_A = "session_A" # Older session
SESSION_ID_B = "session_B" # Current session

# --- Step 1: Simulate an OLD session (Session A) ---
print("\n--- Simulating Session A (Old) ---")
# User tells us a preference
memory_service.add_message(USER_ID, SESSION_ID_A, "user", "I strictly code in Python, never Java.")
memory_service.promote_memory(
    user_id=USER_ID,
    session_id=SESSION_ID_A,
    summary="User prefers Python over Java for coding tasks.",
    importance=0.9,
    m_type="preference"
)

# --- Step 2: Simulate CURRENT session (Session B) ---
print("\n--- Simulating Session B (Current) ---")
# 1. Ingest User Query
query = "Can you write a sorting algorithm for me?"
memory_service.add_message(USER_ID, SESSION_ID_B, "user", query)

# 2. Retrieve Context (The Algorithm)
retrieved_memories = memory_service.retrieve_context(USER_ID, SESSION_ID_B, query)
recent_history = memory_service.get_recent_messages(SESSION_ID_B)

# 3. Formulate Prompt (Displaying what the LLM would see)
print("\n--- Context for LLM ---")
print("SYSTEM: You are a helpful assistant.")

print("\nRECALLED MEMORIES:")
for mem in retrieved_memories:
    print(f"[{mem['type'].upper()}] (Imp: {mem['importance']}): {mem['summary']}")

print("\nRECENT CHAT HISTORY:")
for msg in recent_history:
    print(f"{msg['role']}: {msg['content']}")

# 4. Generate Response (Mock)
print("\n--- LLM Generation ---")
if any("Python" in m['summary'] for m in retrieved_memories):
    response = "Here is a Bubble Sort implementation in Python (based on your preference)..."
else:
    response = "Here is a Bubble Sort in Java..."
    
print(f"Assistant: {response}")

# 5. Save Assistant Response
memory_service.add_message(USER_ID, SESSION_ID_B, "assistant", response)

# memory_service.clear_memories()