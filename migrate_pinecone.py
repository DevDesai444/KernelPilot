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

def _find_text(metadata: dict) -> str:
    for key in _TEXT_CANDIDATES:
        val = metadata.get(key)
        if val:
            return str(val)
    return ""


def _wait_ready(pc: Pinecone, name: str, timeout: int = 180) -> None:
    logger.info("Waiting for index %r to be ready...", name)
    deadline = time.time() + timeout
    while time.time() < deadline:
        info = pc.describe_index(name)
        if isinstance(info, dict):
            state = (info.get("status") or {}).get("state", "")
        else:
            state = getattr(getattr(info, "status", None), "state", "")
        if state == "Ready":
            logger.info("Index %r is ready.", name)
            return
        time.sleep(4)
    raise TimeoutError(f"Index {name!r} not ready within {timeout}s")


def _list_all_ids(index, namespace: str | None) -> list[str]:
    ids: list[str] = []
    kwargs = {}
    if namespace:
        kwargs["namespace"] = namespace
    try:
        for page in index.list(**kwargs):
            if isinstance(page, list):
                ids.extend(page)
            elif hasattr(page, "__iter__"):
                ids.extend(list(page))
            else:
                ids.append(str(page))
    except Exception as exc:
        logger.error("Failed to list index IDs: %s", exc)
        raise
    return ids


def _fetch_records(index, all_ids: list[str], namespace: str | None, batch: int) -> list[dict]:
    records = []
    for i in range(0, len(all_ids), batch):
        chunk = all_ids[i : i + batch]
        kwargs = {"ids": chunk}
        if namespace:
            kwargs["namespace"] = namespace
        resp = index.fetch(**kwargs)
        vectors = getattr(resp, "vectors", None)
        if vectors is None and isinstance(resp, dict):
            vectors = resp.get("vectors", {})
        if not vectors:
            continue
        for rec_id, vec in vectors.items():
            meta = getattr(vec, "metadata", None)
            if meta is None and isinstance(vec, dict):
                meta = vec.get("metadata", {})
            records.append({"_id": rec_id, "metadata": meta or {}})
        logger.info("  fetched %d / %d", min(i + batch, len(all_ids)), len(all_ids))
    return records


def _build_upsert(record: dict, text_field: str) -> dict | None:
    meta = record["metadata"]
    text = _find_text(meta)
    if not text:
        return None
    doc = {"_id": record["_id"], text_field: text}
    for key in _META_FIELDS:
        if meta.get(key):
            doc[key] = meta[key]
    return doc


def _index_names(pc: Pinecone) -> list[str]:
    names = []
    for item in pc.list_indexes():
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict):
            names.append(item.get("name", ""))
        else:
            names.append(getattr(item, "name", "") or "")
    return [n for n in names if n]


# ── main ─────────────────────────────────────────────────────────────────────

