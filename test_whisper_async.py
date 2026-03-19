"""
Async load test for vLLM Whisper large v3.
Extracts up to 10 MP3 files from аудио.zip and sends them concurrently.
"""

import asyncio
import time
import zipfile
import tempfile
import os
from pathlib import Path

import aiohttp


VLLM_URL = "http://localhost:8000/v1/audio/transcriptions"
MODEL = "openai/whisper-large-v3"
ZIP_PATH = "аудио.zip"
MAX_FILES = 10
LANGUAGE = "ru"  # change to None for auto-detect


def extract_audio_files(zip_path: str, dest_dir: str, max_files: int = 10) -> list[str]:
    """Extract MP3 files from zip, skip __MACOSX junk."""
    extracted = []
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if len(extracted) >= max_files:
                break
            if name.startswith("__MACOSX") or not name.lower().endswith(".mp3"):
                continue
            # Flatten filename (remove path separators)
            safe_name = Path(name).name
            dest_path = os.path.join(dest_dir, safe_name)
            with z.open(name) as src, open(dest_path, "wb") as dst:
                dst.write(src.read())
            extracted.append(dest_path)
            print(f"  Extracted: {safe_name}")
    return extracted


async def transcribe_file(
    session: aiohttp.ClientSession,
    file_path: str,
    index: int,
) -> dict:
    """Send one audio file to vLLM /v1/audio/transcriptions."""
    filename = Path(file_path).name
    t0 = time.perf_counter()

    with open(file_path, "rb") as f:
        audio_bytes = f.read()

    form = aiohttp.FormData()
    form.add_field("model", MODEL)
    form.add_field("response_format", "json")
    if LANGUAGE:
        form.add_field("language", LANGUAGE)
    form.add_field(
        "file",
        audio_bytes,
        filename=filename,
        content_type="audio/mpeg",
    )

    try:
        async with session.post(VLLM_URL, data=form) as resp:
            elapsed = time.perf_counter() - t0
            if resp.status == 200:
                data = await resp.json()
                text = data.get("text", "").strip()
                status = "OK"
            else:
                body = await resp.text()
                text = f"HTTP {resp.status}: {body[:200]}"
                status = "ERR"
    except Exception as e:
        elapsed = time.perf_counter() - t0
        text = str(e)
        status = "ERR"

    return {
        "index": index,
        "file": filename,
        "status": status,
        "elapsed": elapsed,
        "text": text,
    }


async def run_test(files: list[str]):
    """Send all files concurrently and collect results."""
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [
            transcribe_file(session, path, i + 1)
            for i, path in enumerate(files)
        ]
        wall_start = time.perf_counter()
        results = await asyncio.gather(*tasks)
        wall_elapsed = time.perf_counter() - wall_start

    return results, wall_elapsed


def print_results(results: list[dict], wall_elapsed: float):
    print("\n" + "=" * 70)
    print(f"{'#':<4} {'File':<45} {'Status':<6} {'Time':>7}")
    print("-" * 70)
    for r in sorted(results, key=lambda x: x["index"]):
        fname = r["file"][:44]
        print(f"{r['index']:<4} {fname:<45} {r['status']:<6} {r['elapsed']:>6.1f}s")
        if r["status"] == "OK":
            preview = r["text"][:120]
            print(f"     > {preview}")
        else:
            print(f"     ! {r['text']}")
        print()

    ok = sum(1 for r in results if r["status"] == "OK")
    total = len(results)
    avg = sum(r["elapsed"] for r in results) / total if total else 0
    print("=" * 70)
    print(f"Files: {ok}/{total} OK")
    print(f"Wall clock time: {wall_elapsed:.1f}s  (all {total} files in parallel)")
    print(f"Avg per-file time: {avg:.1f}s")
    if wall_elapsed > 0:
        print(f"Throughput: {ok / wall_elapsed:.2f} files/sec")


async def main():
    print(f"Extracting up to {MAX_FILES} files from {ZIP_PATH}...")
    with tempfile.TemporaryDirectory() as tmpdir:
        files = extract_audio_files(ZIP_PATH, tmpdir, MAX_FILES)
        if not files:
            print("No MP3 files found in archive!")
            return
        print(f"\nSending {len(files)} files to vLLM at {VLLM_URL} ...")
        results, wall_elapsed = await run_test(files)

    print_results(results, wall_elapsed)


if __name__ == "__main__":
    asyncio.run(main())
