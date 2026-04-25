#!/usr/bin/env python3
"""
migrate_pinecone.py — Migrate the plain-vector cuda-kernels Pinecone index to a
new integrated-inference index so that text search works natively.

What this does:
  1. Fetches all records (IDs + metadata) from the old plain-vector index
  2. Creates a new serverless index with integrated inference
  3. Upserts records — text only, Pinecone re-embeds automatically
  4. Prints the new index host so you can update .env

Usage:
    python migrate_pinecone.py
    python migrate_pinecone.py --old cuda-kernels --new cuda-kernels-v2
    python migrate_pinecone.py --model multilingual-e5-large
    python migrate_pinecone.py --dry-run

After success, update .env:
    PINECONE_INDEX_NAME=cuda-kernels-v2
    PINECONE_INDEX_HOST=<printed at end>
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rlm.env_loader import load_project_env

load_project_env()

try:
    from pinecone import Pinecone
except ImportError:
    sys.exit("pinecone package not installed. Run: pip install pinecone")

# Fields to try when looking for the main text content in metadata
_TEXT_CANDIDATES = ["chunk_text", "source_code", "text", "content", "body", "code"]

# Metadata fields to preserve in the new index
_META_FIELDS = [
    "title", "source", "source_file",
    "hardware_target", "op_type", "optimization_pattern",
]


# ── helpers ──────────────────────────────────────────────────────────────────

