"""
FastAPI Backend for LLM Performance Estimator

Endpoints:
  GET  /api/models           - list all models
  GET  /api/devices          - list all devices
  POST /api/estimate         - single estimation
  POST /api/compare-devices  - compare all devices for a model
  POST /api/compare-models   - compare all models on a device
  POST /api/batch            - batch estimation (multiple model × device pairs)
  GET  /api/roofline/{device_key} - roofline chart data for a device
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional
import dataclasses
import os

from engine import (
    MODELS, DEVICES, PerfConfig,
    estimate_performance, compare_devices, compare_models,
    get_device_tflops,
)

app = FastAPI(
    title="LLM Performance Estimator API",
    version="1.0.0",
    description="Roofline-based performance estimation for LLM inference",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response Models ────────────────────────────────────────────

class EstimateRequest(BaseModel):
    model: str = Field(..., description="Model key, e.g. 'llama-3-8b'")
    device: str = Field(..., description="Device key, e.g. 'rtx-4090'")
    quant_bits: int = Field(4, ge=2, le=16, description="Quantization bits")
    prompt_len: int = Field(512, ge=1, le=1000000)
    output_len: int = Field(256, ge=1, le=1000000)
    batch_size: int = Field(1, ge=1, le=4096)
    tp: int = Field(1, ge=1, le=64, description="Tensor parallelism GPUs")
    compute_util: float = Field(0.55, ge=0.1, le=1.0)
    bw_util: float = Field(0.70, ge=0.1, le=1.0)
    flash_attn: bool = Field(True, description="Use FlashAttention (reduces memory, higher attention utilization)")


class CompareRequest(BaseModel):
    model: Optional[str] = None
    device: Optional[str] = None
    quant_bits: int = Field(4, ge=2, le=16)
    prompt_len: int = Field(512, ge=1, le=1000000)
    output_len: int = Field(256, ge=1, le=1000000)
    batch_size: int = Field(1, ge=1, le=4096)
    tp: int = Field(1, ge=1, le=64)


class BatchItem(BaseModel):
    model: str
    device: str


class BatchRequest(BaseModel):
    items: list[BatchItem]
    quant_bits: int = Field(4, ge=2, le=16)
    prompt_len: int = Field(512, ge=1, le=1000000)
    output_len: int = Field(256, ge=1, le=1000000)
    batch_size: int = Field(1, ge=1, le=4096)
    tp: int = Field(1, ge=1, le=64)


# ─── Helper ───────────────────────────────────────────────────────────────

def _model_info(key: str, m) -> dict:
    return {
        "key": key,
        "name": m.name,
        "params": m.params,
        "active_params": m.active_params,
        "layers": m.layers,
        "heads": m.heads,
        "kv_heads": m.kv_heads,
        "hidden_dim": m.hidden_dim,
        "head_dim": m.head_dim,
        "inter_dim": m.inter_dim,
        "vocab_size": m.vocab_size,
        "max_ctx": m.max_ctx,
        "moe": m.moe,
        "num_experts": m.num_experts,
        "top_k": m.top_k,
        "arch": m.arch,
    }


def _device_info(key: str, d) -> dict:
    return {
        "key": key,
        "name": d.name,
        "vram": d.vram,
        "bw": d.bw,
        "fp16_tflops": d.fp16_tflops,
        "fp8_tflops": d.fp8_tflops,
        "int8_tflops": d.int8_tflops,
        "vendor": d.vendor,
    }


def _validate_model(key: str):
    if key not in MODELS:
        raise HTTPException(404, f"Unknown model: {key}. Use GET /api/models to list.")


def _validate_device(key: str):
    if key not in DEVICES:
        raise HTTPException(404, f"Unknown device: {key}. Use GET /api/devices to list.")


# ─── Endpoints ────────────────────────────────────────────────────────────

@app.get("/api/models")
def list_models():
    """List all available models with their architecture details."""
    return [_model_info(k, m) for k, m in MODELS.items()]


@app.get("/api/devices")
def list_devices():
    """List all available devices with specs."""
    return [_device_info(k, d) for k, d in DEVICES.items()]


@app.post("/api/estimate")
def api_estimate(req: EstimateRequest):
    """Full performance estimation for a model on a device."""
    _validate_model(req.model)
    _validate_device(req.device)

    config = PerfConfig(compute_util=req.compute_util, bw_util=req.bw_util, flash_attn=req.flash_attn)
    r = estimate_performance(
        req.model, req.device, req.quant_bits,
        req.prompt_len, req.output_len, req.batch_size, req.tp, config
    )
    return dataclasses.asdict(r)


@app.post("/api/compare-devices")
def api_compare_devices(req: CompareRequest):
    """Compare all devices for a given model configuration."""
    if not req.model:
        raise HTTPException(400, "model is required for compare-devices")
    _validate_model(req.model)
    return compare_devices(req.model, req.quant_bits, req.prompt_len, req.output_len, req.batch_size, req.tp)


@app.post("/api/compare-models")
def api_compare_models(req: CompareRequest):
    """Compare all models on a given device."""
    if not req.device:
        raise HTTPException(400, "device is required for compare-models")
    _validate_device(req.device)
    return compare_models(req.device, req.quant_bits, req.prompt_len, req.output_len, req.batch_size, req.tp)


@app.post("/api/batch")
def api_batch(req: BatchRequest):
    """Batch estimation for multiple model × device pairs."""
    results = []
    for item in req.items:
        try:
            _validate_model(item.model)
            _validate_device(item.device)
            r = estimate_performance(
                item.model, item.device, req.quant_bits,
                req.prompt_len, req.output_len, req.batch_size, req.tp,
            )
            results.append({"status": "ok", **dataclasses.asdict(r)})
        except HTTPException as e:
            results.append({"status": "error", "model": item.model, "device": item.device, "error": str(e.detail)})
    return results


@app.get("/api/roofline/{device_key}")
def api_roofline(device_key: str, quant_bits: int = 4):
    """Get roofline data points for a device (useful for charting)."""
    _validate_device(device_key)
    device = DEVICES[device_key]

    raw_tflops = get_device_tflops(device, quant_bits)
    peak_tflops = raw_tflops * 0.55  # default compute util
    peak_bw = device.bw * 0.70       # default bw util

    ridge_point = (peak_tflops * 1e12) / (peak_bw * 1e9)

    # Generate roofline curve data points
    ai_range = [2**i for i in range(-4, 16)]
    roofline = []
    for ai in ai_range:
        mem_bound = ai * peak_bw  # GFLOP/s from memory
        perf = min(mem_bound, peak_tflops * 1000)  # GFLOP/s
        roofline.append({"arithmetic_intensity": ai, "performance_gflops": perf})

    return {
        "device": device.name,
        "peak_tflops": peak_tflops,
        "peak_bw_gbs": peak_bw,
        "ridge_point": ridge_point,
        "roofline": roofline,
    }


# ─── Serve static frontend ───────────────────────────────────────────────

@app.get("/")
def serve_index():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "LLM Performance Estimator API", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8765, reload=True)
