import sqlite3
from dataclasses import dataclass, field
from typing import Any, Dict, List

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
    # Paths used by the threaded pipeline so T4 can open its own SQLite connections
    # (sqlite3 cursors are not thread-safe; T4 creates new connections from these paths)
    sqlite_db_path_stage1: str = field(default="")
    sqlite_db_path_stage2: str = field(default="")
    # FAISS cosine-similarity thresholds for OOD rejection. A sample is rejected
    # when its nearest-neighbour cosine similarity falls BELOW the threshold.
    # A very negative value (e.g. -9999.0) disables the gate (never rejects).
    cosine_threshold_stage1: float = field(default=-9999.0)
    cosine_threshold_stage2: float = field(default=-9999.0)
    # Stage-2 segmentation backend (OpenCVSegmenter or SAM2Segmenter), selected
    # by segmentation.method in the YAML config via segmenters.build_segmenter.
    # Typed as Any to avoid importing the (optionally SAM2-dependent) module here.
    segmenter: Any = field(default=None)
    # In-memory accumulator for inference results. Written once at the end of the
    # run via io_utils.dump_results_json; keyed by ground_truth class name.
    results_dict: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
