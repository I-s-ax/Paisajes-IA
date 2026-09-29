#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = ROOT / "labels.txt"
DINO_MODEL = "facebook/dinov2-base"
CLIP_MODEL = "openai/clip-vit-base-patch32"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
MAX_DECODE_SIDE = 2400


def log(message: str) -> None:
    print(message, flush=True)


def load_labels(path: Path) -> list[tuple[str, str]]:
    labels: list[tuple[str, str]] = []
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


def load_image_safe(path: Path) -> Image.Image:
    try:
        with Image.open(path) as img:
            fmt = (img.format or "").upper()

            if max(img.size) > MAX_DECODE_SIDE and fmt in {"JPEG", "JPG"}:
                img.draft("RGB", (MAX_DECODE_SIDE, MAX_DECODE_SIDE))

            img = ImageOps.exif_transpose(img)

            if max(img.size) > MAX_DECODE_SIDE:
                img.thumbnail(
                    (MAX_DECODE_SIDE, MAX_DECODE_SIDE),
                    Image.Resampling.BILINEAR,
                )

            if img.mode != "RGB":
                img = img.convert("RGB")

            return img.copy()
    except (OSError, UnidentifiedImageError) as exc:
        raise RuntimeError(f"No se pudo abrir {path.name}: {exc}") from exc


def autocast_context(device: str):
    if device.startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


class Models:
    def __init__(self, labels_path: Path, device: str) -> None:
        self.device = device
        self.labels = load_labels(labels_path)

        log(f"Cargando DINOv2 en {device}...")
        self.dino_processor = AutoImageProcessor.from_pretrained(DINO_MODEL)
        self.dino_model = AutoModel.from_pretrained(DINO_MODEL).to(device).eval()

        log(f"Cargando CLIP en {device}...")
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

        log("Modelos cargados. Esperando lotes...")

    def analyze_batch(self, paths: list[Path]) -> tuple[list[dict], list[dict]]:
        images: list[Image.Image] = []
        good_paths: list[Path] = []
        errors: list[dict] = []

        for path in paths:
            try:
                images.append(load_image_safe(path))
                good_paths.append(path)
            except Exception as exc:
                errors.append({"file": path.name, "error": str(exc)})

        if not good_paths:
            return [], errors

        try:
            dino_inputs = self.dino_processor(images=images, return_tensors="pt")
            dino_pixels = dino_inputs["pixel_values"].to(
                self.device, non_blocking=True
            )

            clip_inputs = self.clip_processor(images=images, return_tensors="pt")
            clip_pixels = clip_inputs["pixel_values"].to(
                self.device, non_blocking=True
            )

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
            best_prob_np = best_prob.cpu().numpy()
            best_index_np = best_index.cpu().numpy()

            items: list[dict] = []
            for path, vector, confidence, label_index in zip(
                good_paths,
                dino_np,
                best_prob_np,
                best_index_np,
            ):
                items.append(
                    {
                        "file": path.name,
                        "label": self.labels[int(label_index)][0],
                        "confidence": round(float(confidence), 6),
                        "embedding": np.round(vector, 6).tolist(),
                    }
                )

            return items, errors
        finally:
            for image in images:
                image.close()


def process_job(models: Models, job_dir: Path, results_dir: Path, batch_size: int) -> None:
    job_id = job_dir.name
    result_path = results_dir / f"{job_id}.json"
    temp_result = results_dir / f".{job_id}.json.tmp"

    if result_path.exists():
        shutil.rmtree(job_dir, ignore_errors=True)
        return

    images = sorted(
        [
            path
            for path in job_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS
        ],
        key=lambda path: path.name,
    )

    log(f"[{job_id}] Procesando {len(images)} imágenes...")

    all_items: list[dict] = []
    all_errors: list[dict] = []

    try:
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            try:
                items, errors = models.analyze_batch(batch)
            except torch.cuda.OutOfMemoryError:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                log(
                    f"[{job_id}] GPU sin memoria para lote de {len(batch)}; "
                    "reintentando una por una."
                )
                items = []
                errors = []
                for path in batch:
                    try:
                        one_items, one_errors = models.analyze_batch([path])
                        items.extend(one_items)
                        errors.extend(one_errors)
                    except Exception as exc:
                        errors.append({"file": path.name, "error": str(exc)})

            all_items.extend(items)
            all_errors.extend(errors)
            done = min(start + len(batch), len(images))
            log(f"[{job_id}] {done}/{len(images)}")

        payload = {
            "job_id": job_id,
            "items": all_items,
            "errors": all_errors,
        }
    except Exception as exc:
        payload = {
            "job_id": job_id,
            "items": all_items,
            "errors": all_errors,
            "fatal_error": str(exc),
        }

    temp_result.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )
    temp_result.replace(result_path)

    # Las copias de las fotos se eliminan en cuanto el resultado está listo.
    shutil.rmtree(job_dir, ignore_errors=True)
    log(
        f"[{job_id}] Resultado listo. "
        f"Copias temporales eliminadas: {len(images)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Worker persistente DINOv2 + CLIP para Lightning AI."
    )
    parser.add_argument("--root", default="/tmp/paisajes-ia")
    parser.add_argument("--labels", default=str(DEFAULT_LABELS))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--poll", type=float, default=0.5)
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        raise SystemExit("--batch-size debe ser >= 1.")

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
        log(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        log("Aviso: worker ejecutándose en CPU.")

    root = Path(args.root).expanduser().resolve()
    incoming_dir = root / "incoming"
    results_dir = root / "results"
    incoming_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    models = Models(Path(args.labels).expanduser().resolve(), device)

    while True:
        ready_jobs = sorted(
            [
                path
                for path in incoming_dir.iterdir()
                if path.is_dir() and (path / ".ready").exists()
            ],
            key=lambda path: path.name,
        )

        for job_dir in ready_jobs:
            process_job(models, job_dir, results_dir, args.batch_size)

        if args.once:
            break

        time.sleep(args.poll)


if __name__ == "__main__":
    main()
