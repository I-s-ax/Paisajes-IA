#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import shutil
import sqlite3
import subprocess
import tarfile
import tempfile
import time
import uuid
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


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
        CREATE TABLE IF NOT EXISTS file_hashes (
            source_path TEXT PRIMARY KEY,
            size_bytes INTEGER NOT NULL,
            mtime_ns INTEGER NOT NULL,
            content_hash TEXT NOT NULL
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS groups (
            group_id INTEGER PRIMARY KEY AUTOINCREMENT,
            centroid_json TEXT NOT NULL,
            sample_count INTEGER NOT NULL
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS classifications (
            content_hash TEXT PRIMARY KEY,
            group_id INTEGER NOT NULL,
            label TEXT NOT NULL,
            confidence REAL NOT NULL,
            classified_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS processed_files (
            source_path TEXT PRIMARY KEY,
            content_hash TEXT NOT NULL,
            destination TEXT NOT NULL,
            processed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    con.commit()
    return con


def content_hash_for(con: sqlite3.Connection, path: Path) -> str:
    stat = path.stat()
    source = str(path.resolve())

    row = con.execute(
        """
        SELECT content_hash
        FROM file_hashes
        WHERE source_path=? AND size_bytes=? AND mtime_ns=?
        """,
        (source, stat.st_size, stat.st_mtime_ns),
    ).fetchone()

    if row:
        return str(row[0])

    content_hash = sha256_file(path)
    con.execute(
        """
        INSERT INTO file_hashes(source_path, size_bytes, mtime_ns, content_hash)
        VALUES(?,?,?,?)
        ON CONFLICT(source_path) DO UPDATE SET
            size_bytes=excluded.size_bytes,
            mtime_ns=excluded.mtime_ns,
            content_hash=excluded.content_hash
        """,
        (source, stat.st_size, stat.st_mtime_ns, content_hash),
    )
    con.commit()
    return content_hash


def discover_images(input_dir: Path, output_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in input_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue

        try:
            path.resolve().relative_to(output_dir.resolve())
            continue
        except ValueError:
            pass

        files.append(path)

    return sorted(files, key=lambda p: str(p).lower())


def normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def assign_group(
    con: sqlite3.Connection,
    embedding: list[float],
    threshold: float,
) -> int:
    vector = normalize([float(value) for value in embedding])

    rows = con.execute(
        "SELECT group_id, centroid_json, sample_count FROM groups ORDER BY group_id"
    ).fetchall()

    best_group = None
    best_centroid = None
    best_count = 0
    best_score = -1.0

    for group_id, centroid_json, sample_count in rows:
        centroid = json.loads(centroid_json)
        score = cosine(vector, centroid)
        if score > best_score:
            best_score = score
            best_group = int(group_id)
            best_centroid = centroid
            best_count = int(sample_count)

    if best_group is not None and best_score >= threshold:
        new_centroid = [
            ((old * best_count) + new) / (best_count + 1)
            for old, new in zip(best_centroid, vector)
        ]
        new_centroid = normalize(new_centroid)
        con.execute(
            """
            UPDATE groups
            SET centroid_json=?, sample_count=?
            WHERE group_id=?
            """,
            (json.dumps(new_centroid), best_count + 1, best_group),
        )
        con.commit()
        return best_group

    cur = con.execute(
        "INSERT INTO groups(centroid_json, sample_count) VALUES(?, 1)",
        (json.dumps(vector),),
    )
    con.commit()
    return int(cur.lastrowid)


def get_classification(
    con: sqlite3.Connection,
    content_hash: str,
) -> tuple[int, str, float] | None:
    row = con.execute(
        """
        SELECT group_id, label, confidence
        FROM classifications
        WHERE content_hash=?
        """,
        (content_hash,),
    ).fetchone()
    if not row:
        return None
    return int(row[0]), str(row[1]), float(row[2])


def save_classification(
    con: sqlite3.Connection,
    content_hash: str,
    group_id: int,
    label: str,
    confidence: float,
) -> None:
    con.execute(
        """
        INSERT INTO classifications(content_hash, group_id, label, confidence)
        VALUES(?,?,?,?)
        ON CONFLICT(content_hash) DO UPDATE SET
            group_id=excluded.group_id,
            label=excluded.label,
            confidence=excluded.confidence,
            classified_at=CURRENT_TIMESTAMP
        """,
        (content_hash, group_id, label, confidence),
    )
    con.commit()


def already_processed(
    con: sqlite3.Connection,
    path: Path,
    content_hash: str,
) -> bool:
    row = con.execute(
        """
        SELECT 1 FROM processed_files
        WHERE source_path=? AND content_hash=?
        """,
        (str(path.resolve()), content_hash),
    ).fetchone()
    return row is not None


def safe_name(text: str) -> str:
    chars = []
    for char in text.lower().strip():
        if char.isalnum() or char in {"-", "_"}:
            chars.append(char)
        elif char.isspace():
            chars.append("_")
    return "".join(chars) or "otro"


def unique_destination(folder: Path, source: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    candidate = folder / source.name
    if not candidate.exists():
        return candidate

    index = 1
    while True:
        candidate = folder / f"{source.stem}_{index}{source.suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def destination_folder(
    output_dir: Path,
    organize_by: str,
    group_id: int,
    label: str,
) -> Path:
    group_name = f"GRUPO_{group_id:04d}"
    label_name = safe_name(label)

    if organize_by == "label":
        return output_dir / label_name
    if organize_by == "label-group":
        return output_dir / label_name / group_name
    return output_dir / group_name


def append_history(
    csv_path: Path,
    source: Path,
    content_hash: str,
    group_id: int,
    label: str,
    confidence: float,
    destination: Path,
) -> None:
    new_file = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if new_file:
            writer.writerow(
                [
                    "source",
                    "sha256",
                    "group",
                    "label",
                    "confidence",
                    "destination",
                ]
            )
        writer.writerow(
            [
                str(source),
                content_hash,
                f"GRUPO_{group_id:04d}",
                label,
                f"{confidence:.6f}",
                str(destination),
            ]
        )


def record_processed(
    con: sqlite3.Connection,
    source: Path,
    content_hash: str,
    destination: Path,
) -> None:
    con.execute(
        """
        INSERT INTO processed_files(source_path, content_hash, destination)
        VALUES(?,?,?)
        ON CONFLICT(source_path) DO UPDATE SET
            content_hash=excluded.content_hash,
            destination=excluded.destination,
            processed_at=CURRENT_TIMESTAMP
        """,
        (str(source.resolve()), content_hash, str(destination)),
    )
    con.commit()


def build_ssh_base(args) -> tuple[list[str], str]:
    command = ["ssh", "-o", "BatchMode=yes", "-p", str(args.port)]

    if args.identity:
        command.extend(["-i", str(Path(args.identity).expanduser())])

    for option in args.ssh_option:
        command.extend(["-o", option])

    target = f"{args.user}@{args.host}" if args.user else args.host
    return command, target


def run_ssh(
    ssh_base: list[str],
    target: str,
    remote_command: str,
    *,
    stdin=None,
    capture_output: bool = False,
    timeout: int | None = None,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        ssh_base + [target, remote_command],
        stdin=stdin,
        text=False,
        capture_output=capture_output,
        timeout=timeout,
        check=True,
    )


def upload_job(
    ssh_base: list[str],
    target: str,
    remote_root: str,
    representatives: list[tuple[str, Path]],
) -> tuple[str, dict[str, str]]:
    job_id = f"job-{uuid.uuid4().hex}"
    remote_job = f"{remote_root.rstrip('/')}/incoming/{job_id}"
    mapping: dict[str, str] = {}

    with tempfile.NamedTemporaryFile(suffix=".tar") as temp:
        with tarfile.open(fileobj=temp, mode="w") as archive:
            for index, (content_hash, path) in enumerate(representatives):
                suffix = path.suffix.lower() or ".jpg"
                staged = f"{index:04d}{suffix}"
                mapping[staged] = content_hash
                archive.add(path, arcname=staged, recursive=False)

        temp.flush()
        temp.seek(0)

        remote_command = (
            f"mkdir -p {shlex.quote(remote_job)} && "
            f"tar -xf - -C {shlex.quote(remote_job)} && "
            f"touch {shlex.quote(remote_job + '/.ready')}"
        )
        run_ssh(
            ssh_base,
            target,
            remote_command,
            stdin=temp,
            capture_output=True,
        )

    return job_id, mapping


def wait_result(
    ssh_base: list[str],
    target: str,
    remote_root: str,
    job_id: str,
    timeout_seconds: int,
    poll_seconds: int,
) -> dict:
    result_path = f"{remote_root.rstrip('/')}/results/{job_id}.json"
    loops = max(1, timeout_seconds // max(1, poll_seconds))

    remote_command = (
        f"i=0; while [ $i -lt {loops} ]; do "
        f"if [ -f {shlex.quote(result_path)} ]; then "
        f"cat {shlex.quote(result_path)}; exit 0; fi; "
        f"i=$((i+1)); sleep {int(max(1, poll_seconds))}; "
        "done; exit 124"
    )

    completed = run_ssh(
        ssh_base,
        target,
        remote_command,
        capture_output=True,
        timeout=timeout_seconds + 30,
    )

    payload = json.loads(completed.stdout.decode("utf-8"))

    # Confirmamos recepción y eliminamos únicamente el JSON remoto.
    try:
        run_ssh(
            ssh_base,
            target,
            f"rm -f {shlex.quote(result_path)}",
            capture_output=True,
        )
    except Exception:
        pass

    return payload


def organize_file(
    con: sqlite3.Connection,
    source: Path,
    content_hash: str,
    classification: tuple[int, str, float],
    output_dir: Path,
    organize_by: str,
    move: bool,
    history_path: Path,
) -> None:
    group_id, label, confidence = classification
    folder = destination_folder(output_dir, organize_by, group_id, label)
    destination = unique_destination(folder, source)

    if move:
        shutil.move(str(source), str(destination))
    else:
        shutil.copy2(source, destination)

    record_processed(con, source, content_hash, destination)
    append_history(
        history_path,
        source,
        content_hash,
        group_id,
        label,
        confidence,
        destination,
    )

    action = "MOVIDA" if move else "COPIADA"
    log(
        f"{action}: {source.name} -> "
        f"GRUPO_{group_id:04d} | {label} ({confidence:.1%})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Cliente Termux: envía lotes temporales a Lightning AI, "
            "recibe DINOv2+CLIP y organiza localmente."
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--host", default=os.getenv("PAISAJES_SSH_HOST"))
    parser.add_argument("--user", default=os.getenv("PAISAJES_SSH_USER"))
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PAISAJES_SSH_PORT", "22")),
    )
    parser.add_argument(
        "--identity",
        default=os.getenv("PAISAJES_SSH_KEY"),
    )
    parser.add_argument(
        "--ssh-option",
        action="append",
        default=[],
        help="Opción SSH adicional, por ejemplo ProxyJump=...",
    )
    parser.add_argument(
        "--remote-root",
        default=os.getenv("PAISAJES_REMOTE_ROOT", "/tmp/paisajes-ia"),
    )
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--similarity", type=float, default=0.86)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--poll", type=int, default=2)
    parser.add_argument(
        "--organize-by",
        choices=("group", "label", "label-group"),
        default="group",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Mueve los originales. Sin esta opción, los copia.",
    )
    args = parser.parse_args()

    if not args.host:
        raise SystemExit(
            "Falta --host o la variable PAISAJES_SSH_HOST."
        )
    if args.batch_size < 1:
        raise SystemExit("--batch-size debe ser >= 1.")
    if not 0.0 < args.similarity <= 1.0:
        raise SystemExit("--similarity debe estar entre 0 y 1.")

    input_dir = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()

    if not input_dir.is_dir():
        raise SystemExit(f"No existe la carpeta: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    state_dir = output_dir / ".paisajes-ai"
    state_dir.mkdir(parents=True, exist_ok=True)
    con = open_db(state_dir / "estado.sqlite3")
    history_path = output_dir / "resultados.csv"

    ssh_base, target = build_ssh_base(args)

    log("Comprobando conexión SSH...")
    try:
        run_ssh(
            ssh_base,
            target,
            "printf OK",
            capture_output=True,
            timeout=30,
        )
    except Exception as exc:
        con.close()
        raise SystemExit(f"No se pudo conectar por SSH: {exc}")

    files = discover_images(input_dir, output_dir)
    if not files:
        con.close()
        raise SystemExit("No se encontraron imágenes.")

    log(f"Imágenes encontradas: {len(files)}")
    log(f"Acción local: {'mover' if args.move else 'copiar'}")
    log(f"Similitud de grupos: {args.similarity}")
    log("")

    pending_by_hash: dict[str, list[Path]] = {}

    for index, path in enumerate(files, 1):
        try:
            content_hash = content_hash_for(con, path)
        except FileNotFoundError:
            continue

        if already_processed(con, path, content_hash):
            log(f"[{index}/{len(files)}] {path.name} [ya procesada]")
            continue

        cached = get_classification(con, content_hash)
        if cached is not None:
            log(f"[{index}/{len(files)}] {path.name} [clasificación local]")
            organize_file(
                con,
                path,
                content_hash,
                cached,
                output_dir,
                args.organize_by,
                args.move,
                history_path,
            )
            continue

        pending_by_hash.setdefault(content_hash, []).append(path)

    hashes = list(pending_by_hash)
    if not hashes:
        con.close()
        log("No quedan imágenes nuevas por enviar.")
        return

    total_unique = len(hashes)
    log(f"Imágenes únicas nuevas: {total_unique}")

    try:
        for start in range(0, total_unique, args.batch_size):
            batch_hashes = hashes[start : start + args.batch_size]
            representatives = [
                (content_hash, pending_by_hash[content_hash][0])
                for content_hash in batch_hashes
            ]

            log("")
            log(
                f"Subiendo lote "
                f"{start + 1}-{min(start + len(batch_hashes), total_unique)} "
                f"de {total_unique}..."
            )

            job_id, mapping = upload_job(
                ssh_base,
                target,
                args.remote_root,
                representatives,
            )

            log(f"Esperando resultado de Lightning ({job_id})...")
            payload = wait_result(
                ssh_base,
                target,
                args.remote_root,
                job_id,
                args.timeout,
                args.poll,
            )

            if payload.get("fatal_error"):
                raise RuntimeError(
                    f"Worker remoto: {payload['fatal_error']}"
                )

            for error in payload.get("errors", []):
                staged = error.get("file", "?")
                content_hash = mapping.get(staged)
                original = (
                    pending_by_hash.get(content_hash, [None])[0]
                    if content_hash
                    else None
                )
                name = original.name if original else staged
                log(f"ERROR remoto {name}: {error.get('error', 'desconocido')}")

            for item in payload.get("items", []):
                staged = item.get("file")
                content_hash = mapping.get(staged)
                if not content_hash:
                    log(f"Resultado inesperado: {staged}")
                    continue

                embedding = item.get("embedding")
                label = str(item.get("label", "otro"))
                confidence = float(item.get("confidence", 0.0))

                if not isinstance(embedding, list) or not embedding:
                    log(f"Sin embedding: {staged}")
                    continue

                group_id = assign_group(
                    con,
                    embedding,
                    args.similarity,
                )
                save_classification(
                    con,
                    content_hash,
                    group_id,
                    label,
                    confidence,
                )

                classification = (group_id, label, confidence)
                for source in pending_by_hash[content_hash]:
                    if not source.exists():
                        continue
                    organize_file(
                        con,
                        source,
                        content_hash,
                        classification,
                        output_dir,
                        args.organize_by,
                        args.move,
                        history_path,
                    )

    except KeyboardInterrupt:
        log("")
        log("Interrumpido. El progreso local ya guardado se conservará.")
    finally:
        con.close()

    log("")
    log("Terminado.")
    log(f"Resultados locales: {output_dir}")


if __name__ == "__main__":
    main()
