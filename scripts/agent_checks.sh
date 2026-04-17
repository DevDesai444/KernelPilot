#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"

MODE="${1:-}"
KERNEL="${2:-add_rmsnorm_fp4quant_b128xh2048}"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"

usage() {
  cat <<'EOF'
Usage:
  scripts/agent_checks.sh pinecone [query]
  scripts/agent_checks.sh planner [kernel_name]
  scripts/agent_checks.sh coder [kernel_name] [branch_index]
  scripts/agent_checks.sh sandbox [kernel_name] [beam_width] [rounds]
  scripts/agent_checks.sh tree [kernel_name] [beam_width] [rounds]
  scripts/agent_checks.sh all [kernel_name]

Examples:
  scripts/agent_checks.sh pinecone "rmsnorm vectorized stores register pressure cuda code"
  scripts/agent_checks.sh planner add_rmsnorm_fp4quant_b128xh2048
  scripts/agent_checks.sh coder add_rmsnorm_fp4quant_b128xh2048 1
  scripts/agent_checks.sh sandbox add_rmsnorm_fp4quant_b128xh2048 1 1
  scripts/agent_checks.sh tree add_rmsnorm_fp4quant_b128xh2048 2 2
EOF
}

pinecone_check() {
  local query="${2:-rmsnorm vectorized stores register pressure cuda code}"
  python3 check_pinecone.py --query "$query"
}

planner_check() {
  python3 - "$KERNEL" <<'PY'
import json
import logging
import sys
from pathlib import Path

from run import PROJECT_ROOT, WAFERBENCH_KERNELS
from rlm.env_loader import load_project_env
from rlm.environment import RLMEnvironment
from rlm.engine import RLMEngine

kernel_name = sys.argv[1]
kernel_def = next((item for item in WAFERBENCH_KERNELS if item["name"] == kernel_name), None)
if kernel_def is None:
    raise SystemExit(
        "Unknown kernel: "
        + kernel_name
        + "\nAvailable: "
        + ", ".join(item["name"] for item in WAFERBENCH_KERNELS)
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
load_project_env(Path.cwd())

env = RLMEnvironment(
    kernel_name=kernel_def["name"],
    kernel_src_path=str(PROJECT_ROOT / kernel_def["src"]),
    kernel_type=kernel_def["kernel_type"],
    problem_shape=tuple(kernel_def["shape"]),
)
engine = RLMEngine(env)
try:
    plans = engine.run_decompose()
    print(json.dumps(plans, indent=2))
finally:
    engine.close()
PY
}

coder_check() {
  local branch_index="${3:-1}"
  python3 scripts/coder_sandbox_check.py \
    --kernel "$KERNEL" \
    --branch-index "$branch_index"
}

sandbox_check() {
  local beam_width="${3:-1}"
  local rounds="${4:-1}"
  local log_file="$LOG_DIR/sandbox-${KERNEL}-${TIMESTAMP}.log"

  python3 run.py \
    --kernel "$KERNEL" \
    --beam-width "$beam_width" \
    --rounds "$rounds" \
    --log-level INFO 2>&1 | tee "$log_file"

  echo
  echo "Saved log: $log_file"
  echo "Key lines:"
  rg 'SANDBOX|Planner: generating|Planner produced|route=' "$log_file" || true
}

tree_check() {
  local beam_width="${3:-2}"
  local rounds="${4:-2}"
  local log_file="$LOG_DIR/tree-${KERNEL}-${TIMESTAMP}.log"

  python3 run.py \
    --kernel "$KERNEL" \
    --beam-width "$beam_width" \
    --rounds "$rounds" \
    --log-level INFO 2>&1 | tee "$log_file"

  echo
  echo "Saved log: $log_file"
  echo "Key lines:"
  rg 'SANDBOX|Planner: generating|Planner produced|route=planner_tree|route=fixer_with_rag' "$log_file" || true
}

all_check() {
  pinecone_check pinecone
  echo
  echo "===== PLANNER ====="
  planner_check
  echo
  echo "===== CODER ====="
  coder_check coder "$KERNEL" 1
  echo
  echo "===== SANDBOX ====="
  sandbox_check sandbox "$KERNEL" 1 1
  echo
  echo "===== TREE ====="
  tree_check tree "$KERNEL" 2 2
}

case "$MODE" in
  pinecone)
    pinecone_check "$@"
    ;;
  planner)
    planner_check
    ;;
  coder)
    coder_check "$@"
    ;;
  sandbox)
    sandbox_check "$@"
    ;;
  tree)
    tree_check "$@"
    ;;
  all)
    all_check
    ;;
  *)
    usage
    exit 1
    ;;
esac
