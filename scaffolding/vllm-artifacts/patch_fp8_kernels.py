import sys
from pathlib import Path

# FP8 (W8A8) Strix Halo kernel routing for vLLM.
# Kernels: https://github.com/leonyurko/vllm-fp8-strix-halo-kernel-support
# (the kernel modules — fp8_triton.py etc. — are placed on PYTHONPATH, e.g. /opt/fp8).
#
# gfx1151 has no native FP8 tensor support. This injects a routing shim into vLLM's
# compressed-tensors W8A8-FP8 scaled-mm path that *optionally* uses @leonyurko's fused
# FP8->bf16 Triton dequant-GEMM (fp8_triton.fp8_gemm).
#
# OPT-IN: the shim only routes to the Triton kernel when the env var
# VLLM_STRIX_FP8_TRITON=1 is set at serve time; otherwise it calls the stock
# torch._scaled_mm (upstream behavior unchanged). This keeps the kernels off the
# default path — they override stock/hipBLASLt FP8, require --enforce-eager +
# VLLM_ROCM_USE_AITER=0, and aren't benchmarked — so they're only used when
# consciously activated (per kyuz0's suggestion on #67).
#
# Deliberately a SEPARATE patch file (NOT patch_strix.py) so the FP8 work stays
# independent of the is_integrated memory PR (#66). Surgical call-swap (not a file
# overlay): preserves vLLM's current scale-handling in apply_scaled_mm, only
# redirecting the GEMM call. No-ops if already applied.

SCALED_MM_FB = '''

import os as _os
_VLLM_STRIX_FP8_TRITON = _os.environ.get("VLLM_STRIX_FP8_TRITON") == "1"


def _scaled_mm_fb(A, B, *, out_dtype, scale_a, scale_b, bias=None):
    # Default: stock torch._scaled_mm (upstream behavior, incl. any hipBLASLt FP8).
    # Opt-in (VLLM_STRIX_FP8_TRITON=1): gfx1151 fused FP8->bf16 Triton dequant GEMM
    # from leonyurko/vllm-fp8-strix-halo-kernel-support (fp8_triton on PYTHONPATH),
    # with a bf16 matmul + manual dequant fallback if the kernel is unavailable.
    if not _VLLM_STRIX_FP8_TRITON:
        return torch._scaled_mm(
            A, B, out_dtype=out_dtype, scale_a=scale_a, scale_b=scale_b, bias=bias
        )
    try:
        from fp8_triton import fp8_gemm
        return fp8_gemm(A.contiguous(), B, scale_a, scale_b, out_dtype, bias)
    except Exception:
        o = (A.to(torch.bfloat16) @ B.to(torch.bfloat16)).to(torch.float32)
        sa = scale_a.to(torch.float32).reshape(-1)
        sb = scale_b.to(torch.float32).reshape(-1)
        o = o * (sa if sa.numel() == 1 else sa.view(-1, 1))
        o = o * (sb if sb.numel() == 1 else sb.view(1, -1))
        if bias is not None:
            o = o + bias.to(torch.float32)
        return o.to(out_dtype)
'''


def patch_fp8():
    print("Applying Strix Halo FP8 Triton kernel routing to vLLM (opt-in via VLLM_STRIX_FP8_TRITON)...")
    p = Path('vllm/model_executor/kernels/linear/scaled_mm/pytorch.py')
    if not p.exists():
        print(" -> FP8 patch: scaled_mm/pytorch.py not found; skipping (vLLM layout changed?)")
        return
    txt = p.read_text()
    if '_scaled_mm_fb' in txt:
        print(" -> FP8 patch: already applied; skipping")
        return
    n = txt.count('output = torch._scaled_mm(')
    if n == 0:
        print(" -> FP8 patch: no 'output = torch._scaled_mm(' call sites found; skipping")
        return
    # 1) redirect the apply_scaled_mm GEMM calls through the opt-in shim
    txt = txt.replace('output = torch._scaled_mm(', 'output = _scaled_mm_fb(')
    # 2) inject the routing shim at module scope (reads the env var once at import)
    txt = txt + SCALED_MM_FB
    p.write_text(txt)
    print(f" -> FP8 patch: routed {n} scaled_mm call site(s) via _scaled_mm_fb (opt-in: VLLM_STRIX_FP8_TRITON=1)")
    print("Successfully patched vLLM for Strix Halo FP8 kernels (opt-in).")


if __name__ == '__main__':
    patch_fp8()
