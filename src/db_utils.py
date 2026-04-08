import sqlite3
from typing import Any, Optional


def get_label_from_db(cursor: Any, faiss_id: int) -> Optional[str]:
    """
    Queries the SQLite database for the string label matching a FAISS nearest-neighbor ID.

    Args:
        cursor (Any): Active SQLite database cursor.
        faiss_id (int): Integer ID returned by the FAISS nearest-neighbor search.

    Returns:
        Optional[str]: The string label corresponding to the ID if found, otherwise None.
    """
    cursor.execute(
        "SELECT label FROM feature_mappings WHERE faiss_id=?", (int(faiss_id),)
    )
    row = cursor.fetchone()
    return row[0] if row else None
