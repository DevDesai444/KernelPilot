"""
test_llm_picks.py — Test whether the LLM can pick the right strategies for each kernel.

Two modes:
  menu:     LLM picks 4 from a predefined list of 11 strategies
  freeform: LLM proposes 4 optimizations from scratch, no menu at all

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python3 test_llm_picks.py --model sonnet                    # menu mode (default)
    python3 test_llm_picks.py --model sonnet --mode freeform    # no menu, open-ended
    python3 test_llm_picks.py --model all --mode freeform       # compare all 3 freeform
"""

import argparse
import json
import re
import time
from pathlib import Path

import anthropic

PROJECT_ROOT = Path(__file__).parent

# ── Model IDs ────────────────────────────────────────────────────────────────

MODELS = {
    "haiku":  "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
    "opus":   "claude-opus-4-6",
}

# ── Strategy descriptions (for menu mode) ────────────────────────────────────

STRATEGY_DESCRIPTIONS = {
    "vectorize_loads": {
        "desc": "Replace scalar loads with 128-bit float4/uint4 vectorized transactions",
        "applicable": ["add_rmsnorm", "silu_mul", "nvfp4_quantize"],
    },
    "tma_prefetch": {
        "desc": "Use Blackwell TMA engine for async bulk copy with double buffering. "
                "Best for 2D/3D tiled access patterns, NOT simple linear access",
        "applicable": [],
    },
    "warp_reduction": {
        "desc": "Replace __syncthreads block reductions with __shfl_xor_sync warp shuffles",
        "applicable": ["add_rmsnorm"],
    },
    "fuse_passes": {
        "desc": "Combine multiple global memory passes into a single kernel loop. "
                "Only useful when kernel reads the same data in multiple passes",
        "applicable": ["add_rmsnorm"],
    },
    "register_tiling": {
        "desc": "Process multiple elements per thread using register arrays for ILP",
        "applicable": ["add_rmsnorm", "silu_mul"],
    },
    "async_pipeline": {
        "desc": "Overlap memory and compute using cp.async with double buffering. "
                "Best for kernels with loop-carried dependencies to overlap",
        "applicable": [],
    },
    "fp4_lut": {
        "desc": "Replace arithmetic FP4 quantization with a lookup table. "
                "FP4 has only 16 possible output values — a LUT eliminates all quantize math",
        "applicable": ["add_rmsnorm", "silu_mul", "nvfp4_quantize"],
    },
    "fast_math_expf": {
        "desc": "Replace slow expf/logf libcalls (~20 cycles) with hardware SFU intrinsics "
                "__expf/__logf (~4 cycles). Precision loss invisible when output goes to FP4",
        "applicable": ["silu_mul"],
    },
    "thread_coarsening": {
        "desc": "Increase work per thread (multiple quant blocks or rows per thread). "
                "Amortizes thread launch overhead when per-thread work is too small",
        "applicable": ["add_rmsnorm", "silu_mul", "nvfp4_quantize"],
    },
    "ldg_readonly": {
        "desc": "Route read-only inputs through L1 texture cache using __ldg(). "
                "Uses a separate cache path, freeing normal L1 for read-write data",
        "applicable": ["add_rmsnorm", "silu_mul", "nvfp4_quantize"],
    },
    "vectorized_stores": {
        "desc": "Replace byte-by-byte packed FP4 output stores with uint2/uint4 writes. "
                "Reduces store instruction count by 8x",
        "applicable": ["nvfp4_quantize", "silu_mul", "add_rmsnorm"],
    },
}

# ── Kernel sources ────────────────────────────────────────────────────────────

KERNELS = {
    "add_rmsnorm": PROJECT_ROOT / "kernels" / "reference" / "add_rmsnorm.cu",
    "silu_mul":    PROJECT_ROOT / "kernels" / "reference" / "silu_mul.cu",
    "nvfp4_quantize": PROJECT_ROOT / "kernels" / "reference" / "nvfp4_quantize.cu",
}

# ── Prompts ───────────────────────────────────────────────────────────────────

def build_strategy_menu(kernel_type: str) -> str:
    """Build a formatted strategy menu showing all strategies."""
    lines = []
    for name, info in STRATEGY_DESCRIPTIONS.items():
        applicable = info["applicable"]
        applies = "YES" if (not applicable or kernel_type in applicable) else "NO (not applicable to this kernel type)"
        lines.append(f"  - {name}: {info['desc']}")
        lines.append(f"    Applicable to this kernel: {applies}")
    return "\n".join(lines)


def build_menu_prompt(kernel_src: str, kernel_type: str) -> str:
    menu = build_strategy_menu(kernel_type)
    return f"""\
You are a CUDA optimization expert. Select exactly 4 strategies for this kernel.

Kernel type: {kernel_type}
Target: NVIDIA B200 (Blackwell, sm_100a)

```cuda
{kernel_src}
```

Available strategies:
{menu}

RULES:
- Pick only strategies marked "Applicable: YES"
- Pick strategies that match what this kernel actually does
- Do NOT explain your reasoning
- Respond with ONLY a JSON array, nothing else

Example response format:
["strategy_a", "strategy_b", "strategy_c", "strategy_d"]

Your response (JSON array only):"""


def build_freeform_prompt(kernel_src: str, kernel_type: str) -> str:
    return f"""\
You are a CUDA optimization expert. Analyze this kernel and propose exactly 4
optimization techniques that would give the biggest speedup.

You are NOT limited to any predefined list. Propose whatever CUDA optimizations
you think are most impactful for THIS SPECIFIC kernel. Be specific and concrete.

Kernel type: {kernel_type}
Target: NVIDIA B200 (Blackwell, sm_100a, 8 TB/s HBM3e, 142 SMs)

```cuda
{kernel_src}
```

For each optimization, give a short name and one-line description of what to do.

Return as a JSON array of objects, most impactful first:
[
  {{"name": "short_name", "what": "one line description of the concrete change"}},
  {{"name": "short_name", "what": "one line description of the concrete change"}},
  {{"name": "short_name", "what": "one line description of the concrete change"}},
  {{"name": "short_name", "what": "one line description of the concrete change"}}
]

Respond with ONLY the JSON array, nothing else."""


# ── LLM call ─────────────────────────────────────────────────────────────────

def call_llm(client: anthropic.Anthropic, prompt: str, model_id: str) -> tuple:
    """Call the LLM and return (parsed, raw_response, tokens_in, tokens_out, latency)."""
    t0 = time.time()
    response = client.messages.create(
        model=model_id,
        max_tokens=512,
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}],
    )
    latency = time.time() - t0
    text = response.content[0].text
    tokens_in = response.usage.input_tokens
    tokens_out = response.usage.output_tokens
    return text, tokens_in, tokens_out, latency


def parse_menu_response(text: str) -> list:
    """Parse a JSON array of strategy name strings."""
    json_match = re.search(r'\[.*?\]', text, re.DOTALL)
    if json_match:
        try:
            raw = json.loads(json_match.group())
            return [s for s in raw if isinstance(s, str)]
        except (json.JSONDecodeError, TypeError):
            pass
    return []


def parse_freeform_response(text: str) -> list:
    """Parse a JSON array of {name, what} objects."""
    json_match = re.search(r'\[.*?\]', text, re.DOTALL)
    if json_match:
        try:
            raw = json.loads(json_match.group())
            return [item for item in raw if isinstance(item, dict) and "name" in item]
        except (json.JSONDecodeError, TypeError):
            pass
    return []


# ── Menu mode ─────────────────────────────────────────────────────────────────

def run_menu_test(client: anthropic.Anthropic, model_name: str, model_id: str):
    """Menu mode: pick 4 from predefined list."""
    print(f"\n{'='*70}")
    print(f"  MODE: MENU | MODEL: {model_name} ({model_id})")
    print(f"{'='*70}")

    pricing = {
        "haiku":  {"in": 0.25, "out": 1.25},
        "sonnet": {"in": 3.0,  "out": 15.0},
        "opus":   {"in": 15.0, "out": 75.0},
    }
    price = pricing.get(model_name, {"in": 3.0, "out": 15.0})
    total_cost = 0.0

    for kernel_type, src_path in KERNELS.items():
        kernel_src = src_path.read_text()
        prompt = build_menu_prompt(kernel_src, kernel_type)

        print(f"\n  --- {kernel_type} ---")
        text, tok_in, tok_out, latency = call_llm(client, prompt, model_id)
        cost = (tok_in * price["in"] + tok_out * price["out"]) / 1_000_000
        total_cost += cost

        picks = parse_menu_response(text)
        print(f"  Picked:   {picks}")
        print(f"  Tokens:   in={tok_in} out={tok_out}  Cost: ${cost:.4f}  Latency: {latency:.1f}s")

        if not picks:
            print(f"  RAW:      {text[:200]}")

    print(f"\n  Total cost: ${total_cost:.4f}")
    return total_cost


# ── Free-form mode ────────────────────────────────────────────────────────────

def run_freeform_test(client: anthropic.Anthropic, model_name: str, model_id: str):
    """Free-form mode: LLM proposes optimizations from scratch."""
    print(f"\n{'='*70}")
    print(f"  MODE: FREE-FORM | MODEL: {model_name} ({model_id})")
    print(f"{'='*70}")

    pricing = {
        "haiku":  {"in": 0.25, "out": 1.25},
        "sonnet": {"in": 3.0,  "out": 15.0},
        "opus":   {"in": 15.0, "out": 75.0},
    }
    price = pricing.get(model_name, {"in": 3.0, "out": 15.0})
    total_cost = 0.0

    for kernel_type, src_path in KERNELS.items():
        kernel_src = src_path.read_text()
        prompt = build_freeform_prompt(kernel_src, kernel_type)

        print(f"\n  --- {kernel_type} ---")
        text, tok_in, tok_out, latency = call_llm(client, prompt, model_id)
        cost = (tok_in * price["in"] + tok_out * price["out"]) / 1_000_000
        total_cost += cost

        strategies = parse_freeform_response(text)

        if strategies:
            for i, s in enumerate(strategies, 1):
                name = s.get("name", "?")
                what = s.get("what", "?")
                print(f"  {i}. {name:30s} — {what}")
        else:
            print(f"  PARSE FAILED. Raw response:")
            print(f"  {text[:500]}")

        print(f"  Tokens:   in={tok_in} out={tok_out}  Cost: ${cost:.4f}  Latency: {latency:.1f}s")

    print(f"\n  Total cost: ${total_cost:.4f}")
    return total_cost


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test LLM strategy selection")
    parser.add_argument("--model", default="haiku",
                        choices=["haiku", "sonnet", "opus", "all"],
                        help="Which model to test (default: haiku)")
    parser.add_argument("--mode", default="menu",
                        choices=["menu", "freeform", "both"],
                        help="menu=predefined list, freeform=open-ended, both=run both")
    args = parser.parse_args()

    client = anthropic.Anthropic()
    models_to_test = MODELS.items() if args.model == "all" else [(args.model, MODELS[args.model])]

    for name, model_id in models_to_test:
        if args.mode in ("menu", "both"):
            run_menu_test(client, name, model_id)
        if args.mode in ("freeform", "both"):
            run_freeform_test(client, name, model_id)


if __name__ == "__main__":
    main()
