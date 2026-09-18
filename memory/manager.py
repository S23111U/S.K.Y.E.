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
        # WAL mode: semantics.db is now read/written from two separate
        # processes (the main server and the MCP tool-execution process),
        # not just multiple threads in one process — WAL is sqlite's
        # standard answer for concurrent multi-process access without
        # "database is locked" errors.
        cursor.execute("PRAGMA journal_mode=WAL")
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
        # Added after the table already existed with data, so it's a guarded
        # migration rather than part of CREATE TABLE. NULL = still active;
        # a timestamp means supersede_similar() judged it contradicted by a
        # later correction. Rows are marked, never deleted, on purpose — an
        # automated nightly judgment call should stay inspectable/reversible
        # rather than silently destructive.
        existing_columns = {row[1] for row in cursor.execute("PRAGMA table_info(memories)")}
        if "superseded_at" not in existing_columns:
            cursor.execute("ALTER TABLE memories ADD COLUMN superseded_at TEXT DEFAULT NULL")
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
        rows = self.conn.execute(
            "SELECT content, embedding FROM memories WHERE superseded_at IS NULL"
        ).fetchall()
        t2 = time.time()

        if not rows:
            print(f"[search] encode {(t1-t0)*1000:.0f}ms | fetch {(t2-t1)*1000:.0f}ms | "
                f"score 0ms | rows 0")
            return []

        # Vectorized: one BLAS matrix-vector product for every row's cosine
        # similarity at once, instead of a Python loop doing one dot product
        # per row. Same formula, same result — this only ever gets slower as
        # semantics.db grows, and nightly ingestion now actually runs.
        contents = [r[0] for r in rows]
        embeddings = np.frombuffer(
            b"".join(r[1] for r in rows), dtype=np.float32
        ).reshape(len(rows), -1)
        norms = np.linalg.norm(embeddings, axis=1) + 1e-9
        semantic_scores = (embeddings @ query_embedding) / (norms * q_norm)

        keywords = query.lower().split()
        results = []
        for content, semantic in zip(contents, semantic_scores):
            match_count = sum(1 for w in keywords if w in content.lower())
            keyword = match_count / len(keywords) if keywords else 0.0
            results.append(((float(semantic) * 0.7) + (keyword * 0.3), content))
        t3 = time.time()

        print(f"[search] encode {(t1-t0)*1000:.0f}ms | fetch {(t2-t1)*1000:.0f}ms | "
            f"score {(t3-t2)*1000:.0f}ms | rows {len(rows)}")

        results.sort(key=lambda x: x[0], reverse=True)
        return [c for s, c in results[:top_k] if s >= min_score]

    def supersede_similar(self, claim_text, threshold=0.85):
        """Marks existing memories that closely match a now-contradicted claim
        as superseded, so search() stops surfacing them, without deleting the
        row. `threshold` is deliberately well above search()'s own 0.35
        relevance bar — this runs unsupervised from the nightly consolidation
        pass, so it should only ever catch a near-restatement of the same
        claim, not merely-related memories that are still correct.

        Returns the number of rows marked, for the caller to log.
        """
        claim_embedding = self.model.encode(claim_text).astype(np.float32)
        c_norm = np.linalg.norm(claim_embedding) + 1e-9
        rows = self.conn.execute(
            "SELECT id, content, embedding FROM memories WHERE superseded_at IS NULL"
        ).fetchall()
        if not rows:
            return 0

        ids = [r[0] for r in rows]
        contents = [r[1] for r in rows]
        embeddings = np.frombuffer(
            b"".join(r[2] for r in rows), dtype=np.float32
        ).reshape(len(rows), -1)
        norms = np.linalg.norm(embeddings, axis=1) + 1e-9
        scores = (embeddings @ claim_embedding) / (norms * c_norm)

        matched = [
            (row_id, content)
            for row_id, content, score in zip(ids, contents, scores)
            if score >= threshold
        ]

        for row_id, content in matched:
            self.conn.execute(
                "UPDATE memories SET superseded_at = CURRENT_TIMESTAMP WHERE id = ?", (row_id,)
            )
            print(f"[memory] superseded (matched correction '{claim_text}'): {content[:80]}")
        if matched:
            self.conn.commit()
        return len(matched)

    def get_persistent_profile(self):
        if os.path.exists(self.profile_path):
            try:
                with open(self.profile_path, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}