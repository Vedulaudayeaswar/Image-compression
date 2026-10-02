from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import time
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, HTMLResponse
from pydantic import BaseModel, Field

from backend.config import ARTIFACT_DIR, settings
from backend.experiments.store import get, list_experiments
from backend.experiments.report import render_html
from backend.models.foundation import registry
from backend.pipeline import CompressionConfig, compress, decompress, estimator, refiner, unwrap_zip
from backend.training.runner import TrainingConfig, TrainingRunner
from backend.utils.images import ALLOWED_TYPES


app = FastAPI(title="DiffusionCompress", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_methods=["*"], allow_headers=["*"])
jobs: dict[str, dict[str, Any]] = {}
subscribers: dict[str, set[WebSocket]] = defaultdict(set)
trainer = TrainingRunner(estimator, refiner)


class TrainingRequest(BaseModel):
    dataset_path: str
    validation_dataset_path: str | None = None
    iterations: int = Field(default=1_000, ge=1, le=300_000)
    batch_size: int = Field(default=2, ge=1, le=32)
    crop_size: int = Field(default=128, ge=32, le=1024)
    learning_rate: float = Field(default=1e-4, gt=0, le=0.1)
    target_bitrates: list[float] = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
    lambda_latent: float = Field(default=1.0, ge=0)
    lambda_pixel: float = Field(default=1.0, ge=0)
    lambda_perceptual: float = Field(default=0.1, ge=0)
    alpha: float = Field(default=1.0, ge=0, le=1)
    validation_every: int = Field(default=50, ge=1)
    validation_samples: int = Field(default=24, ge=1)
    patience: int = Field(default=8, ge=1)
    checkpoint_every: int = Field(default=50, ge=1)
    channels: int = Field(default=128, ge=16, le=256)
    residual_blocks: int = Field(default=8, ge=1, le=16)
    seed: int = 42
    device: str = "auto"
    mode: str = "demo"


class TrainingControlRequest(BaseModel):
    job_id: str


async def _broadcast(job_id: str) -> None:
    event = jobs[job_id]
    for socket in list(subscribers[job_id]):
        try: await socket.send_json(event)
        except Exception: subscribers[job_id].discard(socket)


def _run_job(job_id: str, data: bytes, filename: str, config: CompressionConfig) -> None:
    def update(stage: str, progress: int) -> None:
        jobs[job_id].update({"stage": stage, "progress": progress})
    try:
        jobs[job_id].update({"status": "running", "stage": "Starting", "progress": 1})
        jobs[job_id]["result"] = compress(data, filename, config, update)
        jobs[job_id]["status"] = "completed"
    except Exception as error:
        jobs[job_id].update({"status": "failed", "error": str(error)})


@app.get("/api/health")
def health() -> dict[str, str]: return {"status": "ok"}


@app.get("/api/model/status")
def model_status() -> dict[str, Any]: return asdict(registry.status())


@app.get("/api/system/status")
def system_status() -> dict[str, Any]:
    import torch
    cuda = torch.cuda.is_available()
    vram = {"allocated_bytes": torch.cuda.memory_allocated(), "reserved_bytes": torch.cuda.memory_reserved()} if cuda else None
    return {"cuda_available": cuda, "device": str(registry.device()), "vram": vram, "gpu_utilization": "not_available", "gpu_temperature": "not_available", "cpu_warning": "CPU mode is intended for experimentation and may be significantly slower." if not cuda else None, "max_upload_bytes": settings.max_upload_bytes}


@app.post("/api/compress")
async def create_compression(background_tasks: BackgroundTasks, image: UploadFile = File(...), mode: str = Form("demo"), bitrate: float = Form(0.30), lambda_value: float = Form(10.0), steps: int = Form(4), resolution: int = Form(256), device: str = Form("auto"), reconstruction_mode: str = Form("baseline"), refinement_alpha: float = Form(0.35)) -> dict[str, str]:
    if image.content_type not in ALLOWED_TYPES: raise HTTPException(415, "Only PNG, JPEG, and WEBP images are supported.")
    data = await image.read()
    if not data or len(data) > settings.max_upload_bytes: raise HTTPException(413, "Image is empty or exceeds the configured 15 MB limit.")
    safe_filename = Path(image.filename or "upload").name
    job_id = f"job-{int(time.time())}-{len(jobs)+1}"
    jobs[job_id] = {"id": job_id, "status": "waiting", "stage": "Queued", "progress": 0}
    # Keep decoder-side hallucinated detail conservative; a full-strength
    # correction can amplify artifacts on images unlike the training set.
    config = CompressionConfig(mode=mode, bitrate=bitrate, lambda_value=lambda_value, steps=steps, resolution=resolution or None, device=device, reconstruction_mode=reconstruction_mode, refinement_alpha=min(refinement_alpha, 0.35))
    background_tasks.add_task(_run_job, job_id, data, safe_filename, config)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job(job_id: str) -> dict[str, Any]:
    if job_id not in jobs: raise HTTPException(404, "Job not found.")
    return jobs[job_id]


@app.websocket("/ws/pipeline/{job_id}")
async def pipeline_socket(websocket: WebSocket, job_id: str) -> None:
    await websocket.accept(); subscribers[job_id].add(websocket)
    try:
        while True:
            await websocket.send_json(jobs.get(job_id, {"status": "unknown"}))
            await asyncio.sleep(0.4)
    except WebSocketDisconnect: subscribers[job_id].discard(websocket)


@app.post("/api/decompress")
async def decompress_file(file: UploadFile = File(...)) -> Response:
    filename = (file.filename or "").lower()
    if not (filename.endswith(".dfc") or filename.endswith(".zip")): raise HTTPException(415, "Upload a .dfc file or ZIP container containing one .dfc file.")
    try:
        payload = await file.read()
        if filename.endswith(".zip"): payload = unwrap_zip(payload)
        png, _ = decompress(payload)
    except Exception as error: raise HTTPException(400, f"Cannot decode DFC: {error}") from error
    return Response(png, media_type="image/png", headers={"Content-Disposition": "attachment; filename=reconstructed.png"})


@app.get("/api/experiments")
def experiments() -> list[dict[str, Any]]: return list_experiments()


@app.get("/api/experiments/{experiment_id}")
def experiment(experiment_id: str) -> dict[str, Any]:
    value = get(experiment_id)
    if value is None: raise HTTPException(404, "Experiment not found.")
    return value


@app.get("/api/metrics/{experiment_id}")
def metrics(experiment_id: str) -> dict[str, Any]:
    value = get(experiment_id)
    if value is None: raise HTTPException(404, "Experiment not found.")
    return value["result"]["metrics"]


@app.get("/api/experiments/{experiment_id}/metrics.json")
def metrics_download(experiment_id: str) -> Response:
    import json
    value = get(experiment_id)
    if value is None: raise HTTPException(404, "Experiment not found.")
    return Response(json.dumps(value["result"]["metrics"], indent=2), media_type="application/json", headers={"Content-Disposition": f"attachment; filename={experiment_id}-metrics.json"})


@app.get("/api/experiments/{experiment_id}/report")
def report(experiment_id: str) -> HTMLResponse:
    value = get(experiment_id)
    if value is None: raise HTTPException(404, "Experiment not found.")
    return HTMLResponse(render_html(value), headers={"Content-Disposition": f"attachment; filename={experiment_id}-report.html"})


@app.get("/api/artifacts/{filename}")
def artifact(filename: str) -> FileResponse:
    path = ARTIFACT_DIR / Path(filename).name
    if not path.is_file(): raise HTTPException(404, "Artifact not found.")
    return FileResponse(path, filename=path.name)


@app.post("/api/reconstruct")
async def reconstruct(file: UploadFile = File(...)) -> Response: return await decompress_file(file)

@app.post("/api/train")
def train(request: TrainingRequest) -> dict[str, str]:
    try:
        config = TrainingConfig(
            dataset_path=request.dataset_path, validation_dataset_path=request.validation_dataset_path,
            iterations=request.iterations, batch_size=request.batch_size, crop_size=request.crop_size,
            learning_rate=request.learning_rate, target_bitrates=tuple(request.target_bitrates),
            lambda_latent=request.lambda_latent, lambda_pixel=request.lambda_pixel,
            lambda_perceptual=request.lambda_perceptual, alpha=request.alpha,
            validation_every=request.validation_every, validation_samples=request.validation_samples,
            patience=request.patience, checkpoint_every=request.checkpoint_every, channels=request.channels,
            residual_blocks=request.residual_blocks, seed=request.seed, device=request.device, mode=request.mode,
        )
        return {"job_id": trainer.start(config)}
    except ValueError as error:
        raise HTTPException(400, str(error)) from error


@app.post("/api/train/pause")
def pause_training(request: TrainingControlRequest) -> dict[str, str]:
    try: trainer.pause(request.job_id)
    except KeyError as error: raise HTTPException(404, "Training job not found.") from error
    return {"status": "pause_requested"}


@app.post("/api/train/resume")
def resume_training(request: TrainingControlRequest) -> dict[str, str]:
    try: trainer.resume(request.job_id)
    except KeyError as error: raise HTTPException(404, "Training job not found.") from error
    return {"status": "running"}


@app.post("/api/train/stop")
def stop_training() -> None:
    raise HTTPException(501, "Stop is not safe for an optimizer mid-step; use pause. A recoverable stop/checkpoint mechanism is not implemented.")


@app.get("/api/training/status")
def training_status(job_id: str | None = None) -> dict[str, Any]:
    if job_id:
        value = trainer.get(job_id)
        if value is None: raise HTTPException(404, "Training job not found.")
        return value
    return {"jobs": list(trainer.jobs.values()), "note": "Demo training optimizes P_phi plus a factorized logistic entropy baseline. Foundation-backbone training remains unavailable until actual SD v2.1 weights are loaded."}


@app.websocket("/ws/training/{job_id}")
async def training_socket(websocket: WebSocket, job_id: str) -> None:
    await websocket.accept()
    try:
        while True:
            value = trainer.get(job_id)
            await websocket.send_json(value or {"status": "unknown"})
            if value and value["status"] in {"completed", "failed"}: break
            await asyncio.sleep(0.5)
    finally:
        await websocket.close()
