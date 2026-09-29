#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from client import (
    already_processed,
    assign_group,
    content_hash_for,
    discover_images,
    get_classification,
    open_db,
    organize_file,
    save_classification,
)


def log(message: str) -> None:
    print(message, flush=True)


def call_api(url: str, api_key: str, batch: list[tuple[str, Path]], timeout: int) -> dict:
    command = [
        "curl",
        "--fail-with-body",
        "--silent",
        "--show-error",
        "--max-time",
        str(timeout),
        "-H",
        f"X-API-Key: {api_key}",
    ]

    for staged_name, path in batch:
        command.extend(
            [
                "-F",
                f"files=@{path};filename={staged_name}",
            ]
        )

    command.append(url.rstrip("/") + "/predict")

    completed = subprocess.run(
        command,
        capture_output=True,
        check=False,
    )

    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        stdout = completed.stdout.decode("utf-8", errors="replace").strip()
        raise RuntimeError(stderr or stdout or f"curl terminó con {completed.returncode}")

    return json.loads(completed.stdout.decode("utf-8"))


def check_health(url: str, timeout: int) -> dict:
    completed = subprocess.run(
        [
            "curl",
            "--fail",
            "--silent",
            "--show-error",
            "--max-time",
            str(timeout),
            url.rstrip("/") + "/health",
        ],
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            completed.stderr.decode("utf-8", errors="replace").strip()
            or "No se pudo contactar el servidor."
        )
    return json.loads(completed.stdout.decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Cliente Termux por HTTPS para Paisajes-IA. "
            "Envía lotes a Lightning y organiza los originales localmente."
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--url", default=os.getenv("PAISAJES_API_URL"))
    parser.add_argument("--api-key", default=os.getenv("PAISAJES_API_KEY"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--similarity", type=float, default=0.86)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument(
        "--organize-by",
        choices=("group", "label", "label-group"),
        default="group",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Mueve originales. Sin esta opción, solo los copia.",
    )
    args = parser.parse_args()

    if not args.url:
        raise SystemExit("Falta --url o PAISAJES_API_URL.")
    if not args.api_key:
        raise SystemExit("Falta --api-key o PAISAJES_API_KEY.")
    if args.batch_size < 1:
        raise SystemExit("--batch-size debe ser >= 1.")

    input_dir = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()

    if not input_dir.is_dir():
        raise SystemExit(f"No existe la carpeta: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    state_dir = output_dir / ".paisajes-ai"
    state_dir.mkdir(parents=True, exist_ok=True)
    con = open_db(state_dir / "estado.sqlite3")
    history_path = output_dir / "resultados.csv"

    log("Comprobando servidor de Lightning...")
    try:
        health = check_health(args.url, min(args.timeout, 30))
    except Exception as exc:
        con.close()
        raise SystemExit(f"No se pudo conectar: {exc}")

    log(
        f"Servidor listo: {health.get('ok')} | "
        f"dispositivo: {health.get('device')}"
    )

    files = discover_images(input_dir, output_dir)
    if not files:
        con.close()
        raise SystemExit("No se encontraron imágenes.")

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
        log("No quedan imágenes nuevas.")
        return

    log(f"Imágenes únicas nuevas: {len(hashes)}")

    try:
        for start in range(0, len(hashes), args.batch_size):
            batch_hashes = hashes[start : start + args.batch_size]
            request_batch = []
            staged_to_hash = {}

            for index, content_hash in enumerate(batch_hashes):
                source = pending_by_hash[content_hash][0]
                staged = f"{index:04d}{source.suffix.lower() or '.jpg'}"
                request_batch.append((staged, source))
                staged_to_hash[staged] = content_hash

            log(
                f"Enviando lote "
                f"{start + 1}-{min(start + len(batch_hashes), len(hashes))} "
                f"de {len(hashes)}..."
            )

            payload = call_api(
                args.url,
                args.api_key,
                request_batch,
                args.timeout,
            )

            for error in payload.get("errors", []):
                staged = error.get("file", "?")
                content_hash = staged_to_hash.get(staged)
                source = (
                    pending_by_hash.get(content_hash, [None])[0]
                    if content_hash
                    else None
                )
                name = source.name if source else staged
                log(f"ERROR remoto {name}: {error.get('error', 'desconocido')}")

            for item in payload.get("items", []):
                staged = item.get("file")
                content_hash = staged_to_hash.get(staged)
                if not content_hash:
                    log(f"Resultado inesperado: {staged}")
                    continue

                embedding = item.get("embedding")
                label = str(item.get("label", "otro"))
                confidence = float(item.get("confidence", 0.0))

                if not isinstance(embedding, list) or not embedding:
                    log(f"Sin embedding: {staged}")
                    continue

                group_id = assign_group(con, embedding, args.similarity)
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
        log("Interrumpido. El progreso local guardado se conservará.")
    finally:
        con.close()

    log("")
    log("Terminado.")
    log(f"Resultados locales: {output_dir}")


if __name__ == "__main__":
    main()
