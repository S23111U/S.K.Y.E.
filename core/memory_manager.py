import os
import json
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer

# =========================================================
# [PILLAR 3: HYBRID RAG MEMORY MANAGER]
# =========================================================

class MemoryManager:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.db_path = os.path.join(root_dir, "memory", "semantics.db")
        self.profile_path = os.path.join(root_dir, "memory", "persistent_profile.json")
        
        # Initialize SQLite for Keyword and Metadata storage
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()
        
        # Initialize Embedding Model (lightweight & fast)
        # It will download on first run (approx 200MB)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def _init_db(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                embedding BLOB NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def add_memory(self, text):
        """Chunks are embedded and stored in SQLite."""
        embedding = self.model.encode(text).astype(np.float32).tobytes()
        cursor = self.conn.cursor()
        cursor.execute("INSERT INTO memories (content, embedding) VALUES (?, ?)", (text, embedding))
        self.conn.commit()

    def search(self, query, top_k=3):
        """Hybrid Search: Vector (Semantic) + Keyword."""
        # 1. Vector Search
        query_embedding = self.model.encode(query).astype(np.float32)
        
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, content, embedding FROM memories")
        rows = cursor.fetchall()
        
        results = []
        for row_id, content, emb_bytes in rows:
            emb = np.frombuffer(emb_bytes, dtype=np.float32)
            # Cosine similarity
            score = np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
            
            # 2. Keyword Boost (Simple term matching)
            keywords = query.lower().split()
            match_count = sum(1 for word in keywords if word in content.lower())
            keyword_score = match_count / len(keywords) if keywords else 0
            
            # Combined Score (Hybrid)
            final_score = (score * 0.7) + (keyword_score * 0.3)
            results.append((final_score, content))
            
        results.sort(key=lambda x: x[0], reverse=True)
        return [res[1] for res in results[:top_k]]

    def get_persistent_profile(self):
        """Loads Tier 3 Structured Profile."""
        if os.path.exists(self.profile_path):
            try:
                with open(self.profile_path, "r") as f:
                    return json.load(f)
            except:
                return {}
        return {}
