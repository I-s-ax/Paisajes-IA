#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import io
import os
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from PIL import Image, ImageOps
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor
import uvicorn

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = ROOT / "labels.txt"
DINO_MODEL = "facebook/dinov2-base"
CLIP_MODEL = "openai/clip-vit-base-patch32"
MAX_DECODE_SIDE = 2400

app = FastAPI(title="Paisajes-IA")
MODELS = None
API_KEY = None


def autocast_context(device: str):
    if device.startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def load_labels(path: Path) -> list[tuple[str, str]]:
    labels = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "|" not in line:
            raise ValueError(f"Línea inválida en {path}: {line!r}")
        name, prompt = line.split("|", 1)
        labels.append((name.strip(), prompt.strip()))
    if not labels:
        raise ValueError("No hay etiquetas válidas.")
    return labels


def image_from_bytes(data: bytes) -> Image.Image:
    with Image.open(io.BytesIO(data)) as img:
        img = ImageOps.exif_transpose(img)
        if max(img.size) > MAX_DECODE_SIDE:
            img.thumbnail(
                (MAX_DECODE_SIDE, MAX_DECODE_SIDE),
                Image.Resampling.BILINEAR,
            )
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img.copy()


class Models:
    def __init__(self, labels_path: Path, device: str) -> None:
        self.device = device
        self.labels = load_labels(labels_path)

        print(f"Cargando DINOv2 en {device}...", flush=True)
        self.dino_processor = AutoImageProcessor.from_pretrained(DINO_MODEL)
        self.dino_model = AutoModel.from_pretrained(DINO_MODEL).to(device).eval()

        print(f"Cargando CLIP en {device}...", flush=True)
        self.clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
        self.clip_model = CLIPModel.from_pretrained(CLIP_MODEL).to(device).eval()

        prompts = [prompt for _, prompt in self.labels]
        text_inputs = self.clip_processor(
            text=prompts,
            return_tensors="pt",
            padding=True,
        )
        text_inputs = {
            key: value.to(device)
            for key, value in text_inputs.items()
            if key in {"input_ids", "attention_mask"}
        }

        with torch.inference_mode(), autocast_context(device):
            text_features = self.clip_model.get_text_features(**text_inputs)
            self.text_features = torch.nn.functional.normalize(
                text_features.float(), p=2, dim=1
            )

        print("Modelos listos.", flush=True)

    def analyze(self, uploads: list[tuple[str, bytes]]) -> tuple[list[dict], list[dict]]:
        images = []
        names = []
        errors = []

        for name, data in uploads:
            try:
                images.append(image_from_bytes(data))
                names.append(name)
            except Exception as exc:
                errors.append({"file": name, "error": str(exc)})

        if not images:
            return [], errors

        try:
            dino_inputs = self.dino_processor(images=images, return_tensors="pt")
            clip_inputs = self.clip_processor(images=images, return_tensors="pt")

            dino_pixels = dino_inputs["pixel_values"].to(self.device, non_blocking=True)
            clip_pixels = clip_inputs["pixel_values"].to(self.device, non_blocking=True)

            with torch.inference_mode(), autocast_context(self.device):
                dino_output = self.dino_model(pixel_values=dino_pixels)
                dino_vectors = dino_output.last_hidden_state[:, 0, :]
                dino_vectors = torch.nn.functional.normalize(
                    dino_vectors.float(), p=2, dim=1
                )

                clip_vectors = self.clip_model.get_image_features(
                    pixel_values=clip_pixels
                )
                clip_vectors = torch.nn.functional.normalize(
                    clip_vectors.float(), p=2, dim=1
                )

                logits = (clip_vectors @ self.text_features.T) * 100.0
                probabilities = torch.softmax(logits, dim=1)
                best_prob, best_index = probabilities.max(dim=1)

            dino_np = dino_vectors.cpu().numpy().astype(np.float32)
            probs_np = best_prob.cpu().numpy()
            indices_np = best_index.cpu().numpy()

            items = []
            for name, vector, confidence, label_index in zip(
                names, dino_np, probs_np, indices_np
            ):
                items.append(
                    {
                        "file": name,
                        "label": self.labels[int(label_index)][0],
                        "confidence": round(float(confidence), 6),
                        "embedding": np.round(vector, 6).tolist(),
                    }
                )

            return items, errors
        finally:
            for image in images:
                image.close()


def require_key(x_api_key: str | None) -> None:
    if not API_KEY or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="API key inválida")


@app.get("/health")
def health():
    return {
        "ok": MODELS is not None,
        "device": MODELS.device if MODELS is not None else None,
    }


@app.post("/predict")
async def predict(
    files: list[UploadFile] = File(...),
    x_api_key: str | None = Header(default=None),
):
    require_key(x_api_key)

    if MODELS is None:
        raise HTTPException(status_code=503, detail="Modelos no cargados")

    uploads = []
    for file in files:
        data = await file.read()
        uploads.append((file.filename or "image", data))

    try:
        items, errors = MODELS.analyze(uploads)
    except torch.cuda.OutOfMemoryError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise HTTPException(
            status_code=507,
            detail="GPU sin memoria. Reduce el tamaño del lote.",
        )

    return {"items": items, "errors": errors}


def main() -> None:
    global MODELS, API_KEY

    parser = argparse.ArgumentParser(
        description="Servidor HTTPS/HTTP para Paisajes-IA en Lightning Studio."
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--labels", default=str(DEFAULT_LABELS))
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    args = parser.parse_args()

    API_KEY = os.getenv("PAISAJES_API_KEY")
    if not API_KEY:
        raise SystemExit(
            "Falta PAISAJES_API_KEY. Define una clave antes de iniciar el servidor."
        )

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("Se pidió CUDA pero PyTorch no detecta GPU.")
        device = "cuda"
    elif args.device == "cpu":
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        print("GPU:", torch.cuda.get_device_name(0), flush=True)

    MODELS = Models(Path(args.labels).expanduser().resolve(), device)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
