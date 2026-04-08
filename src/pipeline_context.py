import sqlite3
from dataclasses import dataclass
from typing import Any, Dict

import torch


@dataclass
class PipelineContext:
    """Context object holding models, databases, and global configurations."""

    model_stage1: torch.nn.Module
    model_stage2: torch.nn.Module
    faiss_index_stage1: Any
    faiss_index_stage2: Any
    sql_cursor_stage1: sqlite3.Cursor
    sql_cursor_stage2: sqlite3.Cursor
    config: Dict[str, Any]
    biomass_params: Dict[str, Any]
    pixel_to_mm2_factor: float
