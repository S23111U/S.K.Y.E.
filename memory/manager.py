"""
Long-term memory for SKYE.

This class used to live inside core_v2.py. It was moved here because
core_v2.py loads an 8B model at import time, which meant any script that
wanted memory access (like ingest_history.py) accidentally loaded Llama into
RAM and never used it.

Nothing about the search logic has changed except:
  - INSERT OR IGNORE, so duplicate memories are skipped instead of stacking
  - a relevance threshold, so irrelevant memories are not injected
"""

import json
import os
import sqlite3

import numpy as np
from sentence_transformers import SentenceTransformer
import time

class MemoryManager:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.db_path = os.path.join(root_dir, "memory", "semantics.db")
        self.profile_path = os.path.join(root_dir, "memory", "persistent_profile.json")

        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

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
        # Enforced at the database level so no script can reintroduce duplicates.
        cursor.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_content ON memories(content)"
        )
        self.conn.commit()

    def add_memory(self, text):
        """Store a memory. Silently skips anything already stored."""
        embedding = self.model.encode(text).astype(np.float32).tobytes()
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT OR IGNORE INTO memories (content, embedding) VALUES (?, ?)",
            (text, embedding),
        )
        self.conn.commit()

    def search(self, query, top_k=3, min_score=0.35):
        t0 = time.time()
        query_embedding = self.model.encode(query).astype(np.float32)
        t1 = time.time()
        q_norm = np.linalg.norm(query_embedding) + 1e-9
        rows = self.conn.execute("SELECT content, embedding FROM memories").fetchall()
        t2 = time.time()

        keywords = query.lower().split()
        results = []
        for content, emb_bytes in rows:
            emb = np.frombuffer(emb_bytes, dtype=np.float32)
            semantic = float(np.dot(query_embedding, emb) / (q_norm * (np.linalg.norm(emb) + 1e-9)))
            match_count = sum(1 for w in keywords if w in content.lower())
            keyword = match_count / len(keywords) if keywords else 0.0
            results.append(((semantic * 0.7) + (keyword * 0.3), content))
        t3 = time.time()

        print(f"[search] encode {(t1-t0)*1000:.0f}ms | fetch {(t2-t1)*1000:.0f}ms | "
            f"score {(t3-t2)*1000:.0f}ms | rows {len(rows)}")

        results.sort(key=lambda x: x[0], reverse=True)
        return [c for s, c in results[:top_k] if s >= min_score]

    def get_persistent_profile(self):
        if os.path.exists(self.profile_path):
            try:
                with open(self.profile_path, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}