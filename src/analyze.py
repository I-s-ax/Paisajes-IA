#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import hashlib
import math
import shutil
import sqlite3
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from transformers import (
    AutoImageProcessor,
    AutoModel,
    CLIPModel,
    CLIPProcessor,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = ROOT / "labels.txt"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
DINO_MODEL = "facebook/dinov2-base"
CLIP_MODEL = "openai/clip-vit-base-patch32"
MAX_DECODE_SIDE = 2400

RESERVED_DIRS = {
    "grupos",
    "etiquetas",
    ".paisajes-ai",
    "__pycache__",
}


def log(message: str) -> None:
    print(message, flush=True)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def open_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS file_locations (
            file_path TEXT PRIMARY KEY,
            content_hash TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            mtime_ns INTEGER NOT NULL,
            last_seen_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS features (
            content_hash TEXT NOT NULL,
            model_key TEXT NOT NULL,
            vector BLOB NOT NULL,
            dimension INTEGER NOT NULL,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(content_hash, model_key)
        )
        """
    )
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_file_locations_hash "
        "ON file_locations(content_hash)"
    )
    con.commit()
    return con


def content_hash_for(con: sqlite3.Connection, path: Path) -> tuple[str, bool]:
    stat = path.stat()
    path_str = str(path.resolve())

    row = con.execute(
        """
        SELECT content_hash
        FROM file_locations
        WHERE file_path=? AND size_bytes=? AND mtime_ns=?
        """,
        (path_str, stat.st_size, stat.st_mtime_ns),
    ).fetchone()

    if row:
        con.execute(
            "UPDATE file_locations SET last_seen_at=CURRENT_TIMESTAMP "
            "WHERE file_path=?",
            (path_str,),
        )
        con.commit()
        return str(row[0]), True

    content_hash = sha256_file(path)
    con.execute(
        """
        INSERT INTO file_locations(
            file_path, content_hash, size_bytes, mtime_ns, last_seen_at
        )
        VALUES(?,?,?,?,CURRENT_TIMESTAMP)
        ON CONFLICT(file_path) DO UPDATE SET
            content_hash=excluded.content_hash,
            size_bytes=excluded.size_bytes,
            mtime_ns=excluded.mtime_ns,
            last_seen_at=CURRENT_TIMESTAMP
        """,
        (path_str, content_hash, stat.st_size, stat.st_mtime_ns),
    )
    con.commit()
    return content_hash, False


def load_feature(
    con: sqlite3.Connection,
    content_hash: str,
    model_key: str,
) -> np.ndarray | None:
    row = con.execute(
        "SELECT vector, dimension FROM features "
        "WHERE content_hash=? AND model_key=?",
        (content_hash, model_key),
    ).fetchone()
    if not row:
        return None

    vector = np.frombuffer(row[0], dtype=np.float32).copy()
    if vector.size != int(row[1]):
        return None
    return vector


def save_feature(
    con: sqlite3.Connection,
    content_hash: str,
    model_key: str,
    vector: np.ndarray,
) -> None:
    vector = np.asarray(vector, dtype=np.float32).ravel()
    con.execute(
        """
        INSERT INTO features(
            content_hash, model_key, vector, dimension, updated_at
        )
        VALUES(?,?,?,?,CURRENT_TIMESTAMP)
        ON CONFLICT(content_hash, model_key) DO UPDATE SET
            vector=excluded.vector,
            dimension=excluded.dimension,
            updated_at=CURRENT_TIMESTAMP
        """,
        (content_hash, model_key, vector.tobytes(), int(vector.size)),
    )
    con.commit()


def discover_images(input_dir: Path, output_dir: Path) -> list[Path]:
    files: list[Path] = []
    output_separate = output_dir.resolve() != input_dir.resolve()

    for path in input_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue

        try:
            rel = path.relative_to(input_dir)
        except ValueError:
            rel = None

        if rel and rel.parts and rel.parts[0] in RESERVED_DIRS:
            continue

        if output_separate:
            try:
                path.resolve().relative_to(output_dir.resolve())
                continue
            except ValueError:
                pass

        files.append(path)

    return sorted(files, key=lambda p: str(p).lower())


def load_image_safe(path: Path) -> Image.Image:
    try:
        with Image.open(path) as img:
            original_size = img.size
            fmt = (img.format or "").upper()

            if max(original_size) > MAX_DECODE_SIDE and fmt in {"JPEG", "JPG"}:
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


def batches(items: list[Path], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def autocast_context(device: str):
    if device.startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def normalize_rows(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return array / norms


def extract_dino_batch(
    processor,
    model,
    paths: list[Path],
    device: str,
) -> tuple[list[Path], np.ndarray, list[tuple[Path, str]]]:
    images: list[Image.Image] = []
    good_paths: list[Path] = []
    errors: list[tuple[Path, str]] = []

    for path in paths:
        try:
            images.append(load_image_safe(path))
            good_paths.append(path)
        except Exception as exc:
            errors.append((path, str(exc)))

    if not good_paths:
        return [], np.empty((0, 768), dtype=np.float32), errors

    try:
        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device, non_blocking=True)

        with torch.inference_mode(), autocast_context(device):
            outputs = model(pixel_values=pixel_values)
            vectors = outputs.last_hidden_state[:, 0, :]
            vectors = torch.nn.functional.normalize(vectors.float(), p=2, dim=1)

        matrix = vectors.cpu().numpy().astype(np.float32)
        return good_paths, matrix, errors
    finally:
        for image in images:
            image.close()


def extract_clip_batch(
    processor,
    model,
    paths: list[Path],
    device: str,
) -> tuple[list[Path], np.ndarray, list[tuple[Path, str]]]:
    images: list[Image.Image] = []
    good_paths: list[Path] = []
    errors: list[tuple[Path, str]] = []

    for path in paths:
        try:
            images.append(load_image_safe(path))
            good_paths.append(path)
        except Exception as exc:
            errors.append((path, str(exc)))

    if not good_paths:
        return [], np.empty((0, 512), dtype=np.float32), errors

    try:
        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device, non_blocking=True)

        with torch.inference_mode(), autocast_context(device):
            vectors = model.get_image_features(pixel_values=pixel_values)
            vectors = torch.nn.functional.normalize(vectors.float(), p=2, dim=1)

        matrix = vectors.cpu().numpy().astype(np.float32)
        return good_paths, matrix, errors
    finally:
        for image in images:
            image.close()


def clear_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def ensure_dino_features(
    con: sqlite3.Connection,
    paths: list[Path],
    hashes: dict[Path, str],
    device: str,
    batch_size: int,
) -> tuple[dict[Path, np.ndarray], list[tuple[Path, str]]]:
    model_key = f"dinov2::{DINO_MODEL}"
    vectors: dict[Path, np.ndarray] = {}
    missing: list[Path] = []

    for path in paths:
        vector = load_feature(con, hashes[path], model_key)
        if vector is None:
            missing.append(path)
        else:
            vectors[path] = vector

    log(
        f"DINOv2: {len(vectors)} en caché, "
        f"{len(missing)} por calcular."
    )

    if not missing:
        return vectors, []

    log("Cargando DINOv2...")
    processor = AutoImageProcessor.from_pretrained(DINO_MODEL)
    model = AutoModel.from_pretrained(DINO_MODEL).to(device)
    model.eval()

    errors: list[tuple[Path, str]] = []
    done = 0

    try:
        for batch in batches(missing, batch_size):
            try:
                good_paths, matrix, batch_errors = extract_dino_batch(
                    processor, model, batch, device
                )
            except torch.cuda.OutOfMemoryError:
                clear_gpu()
                if len(batch) == 1:
                    raise
                log(
                    "Memoria GPU insuficiente para el lote; "
                    "reintentando imagen por imagen..."
                )
                good_paths = []
                rows = []
                batch_errors = []
                for path in batch:
                    try:
                        one_paths, one_matrix, one_errors = extract_dino_batch(
                            processor, model, [path], device
                        )
                        good_paths.extend(one_paths)
                        if len(one_matrix):
                            rows.append(one_matrix[0])
                        batch_errors.extend(one_errors)
                    except Exception as exc:
                        batch_errors.append((path, str(exc)))
                matrix = (
                    np.vstack(rows).astype(np.float32)
                    if rows
                    else np.empty((0, 768), dtype=np.float32)
                )

            for path, vector in zip(good_paths, matrix):
                vector = np.asarray(vector, dtype=np.float32)
                save_feature(con, hashes[path], model_key, vector)
                vectors[path] = vector

            errors.extend(batch_errors)
            done += len(batch)
            log(f"  DINOv2 {done}/{len(missing)}")
    finally:
        del model
        del processor
        clear_gpu()

    return vectors, errors


def ensure_clip_features(
    con: sqlite3.Connection,
    paths: list[Path],
    hashes: dict[Path, str],
    device: str,
    batch_size: int,
) -> tuple[dict[Path, np.ndarray], list[tuple[Path, str]]]:
    model_key = f"clip-image::{CLIP_MODEL}"
    vectors: dict[Path, np.ndarray] = {}
    missing: list[Path] = []

    for path in paths:
        vector = load_feature(con, hashes[path], model_key)
        if vector is None:
            missing.append(path)
        else:
            vectors[path] = vector

    log(
        f"CLIP: {len(vectors)} en caché, "
        f"{len(missing)} por calcular."
    )

    log("Cargando CLIP...")
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    model = CLIPModel.from_pretrained(CLIP_MODEL).to(device)
    model.eval()

    errors: list[tuple[Path, str]] = []
    done = 0

    try:
        for batch in batches(missing, batch_size):
            try:
                good_paths, matrix, batch_errors = extract_clip_batch(
                    processor, model, batch, device
                )
            except torch.cuda.OutOfMemoryError:
                clear_gpu()
                if len(batch) == 1:
                    raise
                log(
                    "Memoria GPU insuficiente para el lote CLIP; "
                    "reintentando imagen por imagen..."
                )
                good_paths = []
                rows = []
                batch_errors = []
                for path in batch:
                    try:
                        one_paths, one_matrix, one_errors = extract_clip_batch(
                            processor, model, [path], device
                        )
                        good_paths.extend(one_paths)
                        if len(one_matrix):
                            rows.append(one_matrix[0])
                        batch_errors.extend(one_errors)
                    except Exception as exc:
                        batch_errors.append((path, str(exc)))
                matrix = (
                    np.vstack(rows).astype(np.float32)
                    if rows
                    else np.empty((0, 512), dtype=np.float32)
                )

            for path, vector in zip(good_paths, matrix):
                vector = np.asarray(vector, dtype=np.float32)
                save_feature(con, hashes[path], model_key, vector)
                vectors[path] = vector

            errors.extend(batch_errors)
            done += len(batch)
            log(f"  CLIP {done}/{len(missing)}")

        return vectors, errors
    finally:
        # Se devuelven también processor/model mediante atributos temporales
        # no; las etiquetas se calculan en una segunda carga ligera de texto.
        del model
        del processor
        clear_gpu()


def load_labels(path: Path) -> list[tuple[str, str]]:
    labels: list[tuple[str, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "|" not in line:
            raise ValueError(
                f"Línea inválida en {path}: {line!r}. "
                "Usa nombre|prompt."
            )
        name, prompt = line.split("|", 1)
        labels.append((name.strip(), prompt.strip()))

    if not labels:
        raise ValueError("No hay etiquetas válidas.")
    return labels


def clip_text_vectors(
    labels: list[tuple[str, str]],
    device: str,
) -> np.ndarray:
    log("Calculando etiquetas de texto con CLIP...")
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    model = CLIPModel.from_pretrained(CLIP_MODEL).to(device)
    model.eval()

    prompts = [prompt for _, prompt in labels]

    try:
        inputs = processor(
            text=prompts,
            return_tensors="pt",
            padding=True,
        )
        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
            if key in {"input_ids", "attention_mask"}
        }

        with torch.inference_mode(), autocast_context(device):
            vectors = model.get_text_features(**inputs)
            vectors = torch.nn.functional.normalize(
                vectors.float(), p=2, dim=1
            )

        return vectors.cpu().numpy().astype(np.float32)
    finally:
        del model
        del processor
        clear_gpu()


def semantic_labels(
    image_vectors: np.ndarray,
    text_vectors: np.ndarray,
    labels: list[tuple[str, str]],
) -> list[tuple[str, float]]:
    logits = image_vectors @ text_vectors.T
    logits = logits * 100.0
    logits = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)

    result: list[tuple[str, float]] = []
    for row in probabilities:
        index = int(np.argmax(row))
        result.append((labels[index][0], float(row[index])))
    return result


def choose_cluster_count(
    matrix: np.ndarray,
    max_clusters: int,
) -> int:
    n = len(matrix)

    if n <= 3:
        return 1

    upper = min(
        max_clusters,
        max(2, int(round(math.sqrt(n)))),
        n - 1,
    )

    if upper < 2:
        return 1

    best_k = 2
    best_score = -2.0

    log(f"Buscando número de grupos (2..{upper})...")

    for k in range(2, upper + 1):
        clusterer = MiniBatchKMeans(
            n_clusters=k,
            random_state=42,
            batch_size=min(256, n),
            n_init=5,
        )
        labels = clusterer.fit_predict(matrix)

        try:
            score = silhouette_score(
                matrix,
                labels,
                metric="cosine",
                sample_size=min(1000, n),
                random_state=42,
            )
        except Exception:
            continue

        log(f"  k={k}: silhouette={score:.4f}")

        if score > best_score:
            best_score = float(score)
            best_k = k

    log(f"Grupos elegidos automáticamente: {best_k}")
    return best_k


def cluster_images(
    matrix: np.ndarray,
    clusters: str,
    max_clusters: int,
) -> tuple[np.ndarray, int]:
    n = len(matrix)

    if n == 1:
        return np.zeros(1, dtype=np.int32), 1

    if clusters == "auto":
        k = choose_cluster_count(matrix, max_clusters)
    else:
        k = int(clusters)

    k = max(1, min(k, n))

    if k == 1:
        return np.zeros(n, dtype=np.int32), 1

    clusterer = MiniBatchKMeans(
        n_clusters=k,
        random_state=42,
        batch_size=min(256, n),
        n_init=10,
    )
    return clusterer.fit_predict(matrix), k


def safe_name(text: str) -> str:
    cleaned = []
    for char in text.strip().lower():
        if char.isalnum() or char in {"-", "_"}:
            cleaned.append(char)
        elif char.isspace():
            cleaned.append("_")
    return "".join(cleaned) or "otro"


def unique_destination(folder: Path, source: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    destination = folder / source.name

    if not destination.exists():
        return destination

    index = 1
    while True:
        destination = folder / f"{source.stem}_{index}{source.suffix}"
        if not destination.exists():
            return destination
        index += 1


def place_file(
    source: Path,
    folder: Path,
    move: bool,
    dry_run: bool,
) -> str:
    destination = unique_destination(folder, source)

    if dry_run:
        return str(destination)

    if move:
        shutil.move(str(source), str(destination))
    else:
        shutil.copy2(source, destination)

    return str(destination)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agrupa e identifica paisajes con DINOv2 + CLIP."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--clusters",
        default="auto",
        help="Número de grupos o 'auto'.",
    )
    parser.add_argument(
        "--max-clusters",
        type=int,
        default=12,
        help="Máximo probado cuando --clusters auto.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Imágenes por lote en GPU.",
    )
    parser.add_argument(
        "--labels",
        default=str(DEFAULT_LABELS),
        help="Archivo nombre|prompt para CLIP.",
    )
    parser.add_argument(
        "--organize-by",
        choices=("group", "label", "both", "none"),
        default="group",
    )
    parser.add_argument("--move", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
    )
    args = parser.parse_args()

    input_dir = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    labels_path = Path(args.labels).expanduser().resolve()

    if not input_dir.is_dir():
        raise SystemExit(f"No existe la carpeta: {input_dir}")
    if not labels_path.is_file():
        raise SystemExit(f"No existe labels: {labels_path}")
    if args.batch_size < 1:
        raise SystemExit("--batch-size debe ser >= 1.")
    if args.max_clusters < 2:
        raise SystemExit("--max-clusters debe ser >= 2.")
    if args.move and args.organize_by == "both":
        raise SystemExit(
            "No se puede usar --move con --organize-by both, "
            "porque un original no puede moverse a dos carpetas."
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

    output_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = output_dir / ".paisajes-ai"
    meta_dir.mkdir(parents=True, exist_ok=True)
    db_path = meta_dir / "paisajes.sqlite3"
    con = open_db(db_path)

    files = discover_images(input_dir, output_dir)
    if not files:
        con.close()
        raise SystemExit("No se encontraron imágenes compatibles.")

    log("=== PAISAJES IA ===")
    log(f"Imágenes: {len(files)}")
    log(f"Dispositivo: {device}")
    if device == "cuda":
        log(f"GPU: {torch.cuda.get_device_name(0)}")
    log(f"DINOv2: {DINO_MODEL}")
    log(f"CLIP: {CLIP_MODEL}")
    log(f"Caché: {db_path}")
    log("")

    hashes: dict[Path, str] = {}
    hash_cache_hits = 0

    log("Calculando/reutilizando hashes...")
    for index, path in enumerate(files, 1):
        content_hash, cached = content_hash_for(con, path)
        hashes[path] = content_hash
        if cached:
            hash_cache_hits += 1
        log(
            f"[HASH {index}/{len(files)}] {path.name}"
            + (" [cache]" if cached else "")
        )

    dino_vectors, dino_errors = ensure_dino_features(
        con,
        files,
        hashes,
        device,
        args.batch_size,
    )

    clip_vectors, clip_errors = ensure_clip_features(
        con,
        files,
        hashes,
        device,
        args.batch_size,
    )

    valid_files = [
        path
        for path in files
        if path in dino_vectors and path in clip_vectors
    ]

    if not valid_files:
        con.close()
        raise SystemExit("No se pudo analizar ninguna imagen.")

    labels = load_labels(labels_path)
    text_vectors = clip_text_vectors(labels, device)

    dino_matrix = normalize_rows(
        np.vstack([dino_vectors[path] for path in valid_files])
    )
    clip_matrix = normalize_rows(
        np.vstack([clip_vectors[path] for path in valid_files])
    )

    semantic = semantic_labels(
        clip_matrix,
        text_vectors,
        labels,
    )

    cluster_labels, cluster_count = cluster_images(
        dino_matrix,
        args.clusters,
        args.max_clusters,
    )

    csv_path = output_dir / "resultados.csv"
    errors_by_path: dict[Path, list[str]] = {}

    for path, error in dino_errors + clip_errors:
        errors_by_path.setdefault(path, []).append(error)

    rows = []

    log("")
    log("Organizando resultados...")

    for index, (path, cluster, semantic_result) in enumerate(
        zip(valid_files, cluster_labels, semantic),
        1,
    ):
        label_name, confidence = semantic_result
        group_name = f"GRUPO_{int(cluster) + 1:03d}"
        destinations: list[str] = []

        if args.organize_by in {"group", "both"}:
            folder = output_dir / "grupos" / group_name
            destinations.append(
                place_file(path, folder, args.move, args.dry_run)
            )

        if args.organize_by in {"label", "both"}:
            folder = output_dir / "etiquetas" / safe_name(label_name)
            destinations.append(
                place_file(path, folder, False, args.dry_run)
            )

        log(
            f"[{index}/{len(valid_files)}] {path.name} "
            f"-> {group_name} | {label_name} "
            f"({confidence:.1%})"
        )

        rows.append(
            [
                str(path),
                hashes[path],
                group_name,
                label_name,
                f"{confidence:.6f}",
                " | ".join(destinations),
                "",
            ]
        )

    invalid_paths = [
        path for path in files if path not in valid_files
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "archivo",
                "sha256",
                "grupo",
                "etiqueta",
                "confianza_clip",
                "destino",
                "error",
            ]
        )
        writer.writerows(rows)

        for path in invalid_paths:
            writer.writerow(
                [
                    str(path),
                    hashes.get(path, ""),
                    "",
                    "",
                    "",
                    "",
                    " | ".join(errors_by_path.get(path, ["error desconocido"])),
                ]
            )

    con.close()

    log("")
    log("=== RESUMEN ===")
    log(f"Analizadas: {len(valid_files)}/{len(files)}")
    log(f"Grupos: {cluster_count}")
    log(f"Hashes reutilizados: {hash_cache_hits}/{len(files)}")
    log(f"Errores: {len(invalid_paths)}")
    log(f"CSV: {csv_path}")

    counts: dict[str, int] = {}
    for label, _ in semantic:
        counts[label] = counts.get(label, 0) + 1

    log("Etiquetas:")
    for name, count in sorted(
        counts.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        log(f"  {name}: {count}")

    if args.dry_run:
        log("Modo simulación: no se copiaron ni movieron imágenes.")


if __name__ == "__main__":
    main()
