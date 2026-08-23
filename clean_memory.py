"""
One-time cleanup for memory/semantics.db.

Run from the repo root:  python clean_memory.py

Three things happen:
  1. Duplicate rows are removed (keeping the earliest copy of each).
  2. Junk memories are removed — failed speech recognition, empty greetings.
     These are not facts about you, and when they get retrieved they waste
     prompt space and confuse the model.
  3. A UNIQUE index is added so duplicates can never be inserted again.

Your original file is backed up first. Nothing is destroyed.
"""

import os
import shutil
import sqlite3
import sys

DB = os.path.join("memory", "semantics.db")

# Memories matching these are noise, not knowledge about you.
JUNK = [
    "%Couldn't Understand%",
    "%Can you repeat again%",
    "%[Corrected unique response placeholder]%",
    "%Standing by for your directive%",
    "%All systems functioning within normal parameters%",
    "%Everything is functioning flawlessly%",
]


def main():
    if not os.path.exists(DB):
        sys.exit(f"Can't find {DB}. Run this from the repo root (the jarvis/ folder).")

    backup = DB + ".backup"
    shutil.copy2(DB, backup)
    print(f"Backed up to {backup}\n")

    con = sqlite3.connect(DB)
    before = con.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    # 1. Deduplicate. MIN(id) keeps the oldest copy of each distinct memory.
    con.execute(
        "DELETE FROM memories WHERE id NOT IN "
        "(SELECT MIN(id) FROM memories GROUP BY content)"
    )
    after_dedup = con.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    print(f"Removed {before - after_dedup} duplicate rows  ({before} -> {after_dedup})")

    # 2. Remove junk.
    removed = 0
    for pattern in JUNK:
        cur = con.execute("DELETE FROM memories WHERE content LIKE ?", (pattern,))
        removed += cur.rowcount
    print(f"Removed {removed} junk memories")

    # 3. Prevent recurrence at the database level, so a buggy script can't
    #    reintroduce duplicates even if someone forgets to check.
    con.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_content ON memories(content)")

    con.commit()
    final = con.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    con.execute("VACUUM")          # reclaim disk space
    con.close()

    print(f"\nDone. {before} rows -> {final} rows.")
    print("If anything looks wrong, restore with:")
    print(f"  cp {backup} {DB}")


if __name__ == "__main__":
    main()