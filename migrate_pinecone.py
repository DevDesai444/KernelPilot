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

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Migrate Pinecone plain-vector index to integrated-inference index."
    )
    parser.add_argument("--old", default="cuda-kernels",
                        help="Source index name (default: cuda-kernels)")
    parser.add_argument("--new", default="cuda-kernels-v2",
                        help="Destination index name (default: cuda-kernels-v2)")
    parser.add_argument("--model", default="multilingual-e5-large",
                        help="Pinecone embed model for integrated inference "
                             "(default: multilingual-e5-large)")
    parser.add_argument("--cloud", default="aws",  help="Cloud provider (default: aws)")
    parser.add_argument("--region", default="us-east-1", help="Region (default: us-east-1)")
    parser.add_argument("--namespace", default=None, help="Pinecone namespace (optional)")
    parser.add_argument("--fetch-batch", type=int, default=200,
                        help="IDs per fetch call (default: 200)")
    parser.add_argument("--upsert-batch", type=int, default=50,
                        help="Records per upsert call (default: 50)")
    parser.add_argument("--batch-delay", type=float, default=4.0,
                        help="Seconds to sleep between upsert batches to avoid rate limits (default: 4.0)")
    parser.add_argument("--start-from", type=int, default=0,
                        help="Skip this many records at the start — use to resume after a 429 (default: 0)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch and count only — no writes")
    args = parser.parse_args()

    api_key = os.environ.get("PINECONE_API_KEY", "")
    if not api_key:
        sys.exit("PINECONE_API_KEY is not set. Check your .env file.")

    pc = Pinecone(api_key=api_key)
    text_field = "chunk_text"

    # ── Step 1: fetch all records from old index ──────────────────────────
    logger.info("=== Step 1: Fetching records from %r ===", args.old)
    old_index = pc.Index(args.old)

    logger.info("Listing all IDs...")
    all_ids = _list_all_ids(old_index, args.namespace)
    logger.info("Found %d IDs.", len(all_ids))

    if not all_ids:
        sys.exit(f"No records found in index {args.old!r}. Nothing to migrate.")

    logger.info("Fetching metadata for all records...")
    records = _fetch_records(old_index, all_ids, args.namespace, args.fetch_batch)
    logger.info("Fetched %d records.", len(records))

    upsert_docs = [d for d in (_build_upsert(r, text_field) for r in records) if d]
    skipped = len(records) - len(upsert_docs)
    logger.info("%d records have text — will migrate.", len(upsert_docs))
    if skipped:
        logger.warning("%d records skipped (no text content in metadata).", skipped)

    if args.dry_run:
        logger.info("=== DRY RUN — no writes performed ===")
        if upsert_docs:
            logger.info("Sample record keys: %s", sorted(upsert_docs[0].keys()))
            logger.info("Sample text (first 120 chars): %s",
                        upsert_docs[0].get(text_field, "")[:120])
        return 0

    # ── Step 2: create new index with integrated inference ────────────────
    logger.info("=== Step 2: Creating new index %r ===", args.new)
    existing = _index_names(pc)
    if args.new in existing:
        logger.info("Index %r already exists — skipping creation.", args.new)
    else:
        logger.info("Creating index with model=%r on %s/%s...", args.model, args.cloud, args.region)
        pc.create_index_for_model(
            name=args.new,
            cloud=args.cloud,
            region=args.region,
            embed={
                "model": args.model,
                "field_map": {"text": text_field},
            },
        )
        _wait_ready(pc, args.new)

    new_index = pc.Index(args.new)

    # ── Step 3: upsert records ────────────────────────────────────────────
    remaining = upsert_docs[args.start_from:]
    total = len(upsert_docs)
    if args.start_from:
        logger.info("Resuming from record %d (skipping first %d already upserted).",
                    args.start_from, args.start_from)
    logger.info("=== Step 3: Upserting %d records into %r ===", len(remaining), args.new)

    for i in range(0, len(remaining), args.upsert_batch):
        batch = remaining[i : i + args.upsert_batch]
        # Retry up to 5 times on 429 with exponential backoff
        for attempt in range(5):
            try:
                new_index.upsert_records(args.namespace or "__default__", batch)
                break
            except Exception as exc:
                if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                    wait = 60 * (attempt + 1)
                    logger.warning("Rate limited (429). Waiting %ds before retry %d/5...", wait, attempt + 1)
                    time.sleep(wait)
                else:
                    raise
        done = args.start_from + i + len(batch)
        logger.info("  upserted %d / %d", done, total)
        if i + args.upsert_batch < len(remaining):
            time.sleep(args.batch_delay)

    # ── Step 4: print update instructions ────────────────────────────────
    logger.info("=== Migration complete ===")
    info = pc.describe_index(args.new)
    host = info.get("host") if isinstance(info, dict) else getattr(info, "host", None)

    print("\n" + "=" * 60)
    print("Update your .env with:")
    print(f"  PINECONE_INDEX_NAME={args.new}")
    if host:
        print(f"  PINECONE_INDEX_HOST={host}")
    print("=" * 60 + "\n")

    logger.info("Then run: python check_pinecone.py --query 'fused add rmsnorm fp4'")
    logger.info("Expected query_mode: text (no fallback)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
