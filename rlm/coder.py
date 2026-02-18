from __future__ import annotations


def _infer_kernel_type(launch_signature: str, kernel_code: str) -> str:
    haystack = f"{launch_signature}\n{kernel_code}"
    if "launch_fused_add_rmsnorm_nvfp4" in haystack:
        return "add_rmsnorm"
    if "launch_silu_mul_fp4quant" in haystack:
        return "silu_mul"
    if "launch_nvfp4_quantize_bf16" in haystack:
        return "nvfp4_quantize"
    return "unknown"


def _branch_mentions(text: str, *terms: str) -> bool:
    norm = text.lower()
    return any(term in norm for term in terms)


