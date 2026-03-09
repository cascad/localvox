#!/usr/bin/env python3
"""
Бенчмарк: прогон SOVA датасета через live-transcribe-server-reliable и сравнение с эталоном.

Использование:
  # HuggingFace bond005/sova_rudevices (test split)
  python tools/bench_sova.py --server ws://localhost:9745 --dataset hf:bond005/sova_rudevices --split test --limit 100

  # Локальная папка: manifest.json с полями path, transcription
  python tools/bench_sova.py --server ws://localhost:9745 --dataset local:./sova_test --limit 50

  # Локальная папка: .wav и .txt с одинаковыми именами
  python tools/bench_sova.py --server ws://localhost:9745 --dataset dir:./wavs --limit 50

  # Локальная папка: manifest.jsonl (path, transcription) или manifest.json (массив)
  python tools/bench_sova.py --server ws://localhost:9745 --dataset local:./sova_test --limit 50

Зависимости: pip install websockets jiwer
Опционально: pip install datasets  (для HuggingFace)
             pip install soundfile scipy (для WAV с нестандартной частотой)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import struct
import sys
from pathlib import Path

SAMPLE_RATE = 16000
CHUNK_FRAMES = 512
SOURCE_ID = 0
WAIT_AFTER_SEND_SEC = 15.0
RECV_TIMEOUT_SEC = 30.0
# Тишина после аудио, чтобы VAD сервера сбросил сегмент (vad_silence_sec ~1с, min_chunk_sec 2с)
SILENCE_PADDING_SEC = 2.5


def load_wav_pcm(path: str | Path) -> tuple[bytes, int]:
    """Load WAV as 16-bit mono PCM. Returns (bytes, sample_rate). Resamples to 16kHz if needed."""
    try:
        import wave
    except ImportError:
        try:
            import soundfile as sf
            data, sr = sf.read(path, dtype="int16")
            if sr != SAMPLE_RATE:
                import numpy as np
                from scipy import signal
                data = signal.resample_poly(data, SAMPLE_RATE, sr).astype(np.int16)
            return data.tobytes(), SAMPLE_RATE
        except ImportError:
            pass
        raise RuntimeError("Need wave (stdlib) or soundfile for WAV loading")

    with wave.open(str(path), "rb") as w:
        ch, sw, sr, nf, _, _ = w.getparams()
        if sw != 2 or ch != 1:
            raise ValueError(f"Expected 16-bit mono WAV, got {sw*8}-bit {ch}ch")
        pcm = w.readframes(nf)
    if sr != SAMPLE_RATE:
        import numpy as np
        from scipy import signal
        arr = np.frombuffer(pcm, dtype=np.int16)
        arr = signal.resample_poly(arr, SAMPLE_RATE, sr).astype(np.int16)
        pcm = arr.tobytes()
    return pcm, SAMPLE_RATE


def iter_dataset(dataset_spec: str, split: str | None, limit: int | None):
    """Yield (audio_path, reference_text) from dataset."""
    if dataset_spec.startswith("hf:"):
        name = dataset_spec[3:].strip()
        try:
            from datasets import load_dataset
        except ImportError:
            raise RuntimeError("pip install datasets for HuggingFace support")
        ds = load_dataset(name, split=split or "test")
        # Access raw PyArrow table to avoid Audio feature decode (which requires torchcodec)
        table = ds.data
        col_names = table.column_names
        trans_col_name = "transcription" if "transcription" in col_names else "text"
        if "audio" not in col_names or trans_col_name not in col_names:
            raise ValueError(f"Dataset needs 'audio' and '{trans_col_name}' columns, got: {col_names}")
        audio_col = table.column("audio")
        trans_col = table.column(trans_col_name)
        n = min(limit or len(ds), len(ds))
        for i in range(n):
            audio_val = audio_col[i]
            path = None
            if hasattr(audio_val, "as_py"):
                d = audio_val.as_py()
                if isinstance(d, dict):
                    path = d.get("path")
                    if not path and d.get("bytes"):
                        import tempfile
                        import os
                        fd, path = tempfile.mkstemp(suffix=".wav")
                        try:
                            os.write(fd, d["bytes"])
                        finally:
                            os.close(fd)
                elif isinstance(d, str):
                    path = d
            elif audio_val is not None:
                path = str(audio_val)
            ref_val = trans_col[i]
            ref = ref_val.as_py() if hasattr(ref_val, "as_py") else str(ref_val)
            if path and ref:
                yield path, ref

    elif dataset_spec.startswith("local:") or dataset_spec.startswith("dir:"):
        root = Path(dataset_spec.split(":", 1)[1].strip())
        if not root.is_dir():
            raise FileNotFoundError(f"Directory not found: {root}")

        if dataset_spec.startswith("local:"):
            manifest = root / "manifest.json"
            if not manifest.is_file():
                manifest = root / "manifest.jsonl"
            if manifest.is_file():
                with open(manifest, encoding="utf-8") as f:
                    if manifest.suffix == ".jsonl":
                        items_iter = (json.loads(line.strip()) for line in f if line.strip())
                    else:
                        data = json.load(f)
                        items_iter = data if isinstance(data, list) else [data]
                count = 0
                for data in items_iter:
                    if limit and count >= limit:
                        break
                    path = data.get("path") or data.get("audio_filepath") or data.get("audio")
                    ref = data.get("transcription") or data.get("text") or data.get("reference")
                    if not path or ref is None:
                        continue
                    path = root / path if not Path(path).is_absolute() else Path(path)
                    if path.is_file():
                        count += 1
                        yield str(path), ref
                return
            raise FileNotFoundError(f"manifest.json not found in {root}")

        # dir: — .wav и .txt с одинаковыми именами
        count = 0
        for wav in sorted(root.glob("*.wav")):
            if limit and count >= limit:
                break
            txt = wav.with_suffix(".txt")
            if txt.is_file():
                ref = txt.read_text(encoding="utf-8").strip()
                count += 1
                yield str(wav), ref

    else:
        raise ValueError(f"Unknown dataset spec: {dataset_spec}. Use hf:..., local:..., or dir:...")


async def transcribe_file(ws, wav_path: str, ref: str, verbose: bool) -> str:
    """Stream WAV to server, collect transcript. Returns hypothesis text."""
    pcm, _ = load_wav_pcm(wav_path)
    n_samples = len(pcm) // 2
    for i in range(0, n_samples, CHUNK_FRAMES):
        chunk = pcm[i * 2 : (i + CHUNK_FRAMES) * 2]
        if len(chunk) < CHUNK_FRAMES * 2:
            chunk = chunk + b"\x00" * (CHUNK_FRAMES * 2 - len(chunk))
        msg = bytes([SOURCE_ID]) + chunk
        await ws.send(msg)

    # Тишина после аудио — VAD сбросит сегмент (vad_silence_sec + min_chunk)
    silence_chunk = bytes([SOURCE_ID]) + (b"\x00\x00" * CHUNK_FRAMES)
    n_silence = int(SAMPLE_RATE * SILENCE_PADDING_SEC / CHUNK_FRAMES)
    for _ in range(n_silence):
        await ws.send(silence_chunk)

    segments: list[str] = []
    try:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + WAIT_AFTER_SEND_SEC + RECV_TIMEOUT_SEC
        while loop.time() < deadline:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=RECV_TIMEOUT_SEC)
            except asyncio.TimeoutError:
                break
            if isinstance(msg, bytes):
                continue
            try:
                data = json.loads(msg)
            except json.JSONDecodeError:
                continue
            t = data.get("type")
            if t == "transcript":
                text = data.get("text", "")
                src = data.get("source", 0)
                if src == SOURCE_ID and text:
                    segments.append(text)
                    if verbose:
                        print(f"    segment: {text[:60]}...")
    except Exception as e:
        if verbose:
            print(f"    recv error: {e}")

    hypothesis = " ".join(segments).strip()
    return hypothesis


def normalize_for_wer(text: str) -> str:
    """Lowercase, strip punctuation, collapse spaces — для честного WER."""
    import re
    t = text.lower().strip()
    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"\s+", " ", t)
    return t


def _levenshtein_chars(a: str, b: str) -> int:
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[n][m]


def compute_wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate."""
    ref = normalize_for_wer(reference)
    hyp = normalize_for_wer(hypothesis)
    ref_words = ref.split()
    hyp_words = hyp.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    try:
        import jiwer
        return jiwer.wer(ref, hyp)
    except ImportError:
        n, m = len(ref_words), len(hyp_words)
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
        return dp[n][m] / n


def compute_cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate — мягче к опечаткам и перестановкам слов."""
    ref = normalize_for_wer(reference)
    hyp = normalize_for_wer(hypothesis)
    ref = ref.replace(" ", "")
    hyp = hyp.replace(" ", "")
    if not ref:
        return 0.0 if not hyp else 1.0
    edits = _levenshtein_chars(ref, hyp)
    return edits / len(ref)


async def run_benchmark(
    server_url: str,
    dataset_spec: str,
    split: str | None,
    limit: int | None,
    verbose: bool,
    output_json: str | None,
) -> None:
    try:
        import websockets
    except ImportError:
        print("pip install websockets", file=sys.stderr)
        sys.exit(1)

    ws_url = server_url.replace("http://", "ws://").replace("https://", "wss://")
    if "/" not in ws_url.split("//")[-1]:
        ws_url = ws_url.rstrip("/") + "/"

    items = list(iter_dataset(dataset_spec, split, limit))
    if not items:
        print("No items in dataset", file=sys.stderr)
        sys.exit(1)
    print(f"Dataset: {dataset_spec} (split={split}), {len(items)} items")

    results: list[dict] = []
    total_wer = 0.0
    total_cer = 0.0
    total_ref_words = 0
    total_ref_chars = 0

    for idx, (wav_path, ref) in enumerate(items):
        if verbose:
            print(f"[{idx + 1}/{len(items)}] {Path(wav_path).name}")
        async with websockets.connect(ws_url) as ws:
            await ws.send(json.dumps({"type": "config", "source_count": 1}))
            await ws.send(json.dumps({"type": "recording", "enabled": True}))
            hyp = await transcribe_file(ws, wav_path, ref, verbose)
        wer = compute_wer(ref, hyp)
        cer = compute_cer(ref, hyp)
        ref_words = len(ref.split())
        ref_chars = len(normalize_for_wer(ref).replace(" ", ""))
        total_wer += wer * ref_words
        total_cer += cer * ref_chars if ref_chars > 0 else 0
        total_ref_words += ref_words
        total_ref_chars += ref_chars
        results.append({
            "path": wav_path,
            "reference": ref,
            "hypothesis": hyp,
            "wer": wer,
            "cer": cer,
            "ref_words": ref_words,
        })
        if verbose:
            print(f"  CER={cer:.1%}  ref={ref[:50]}...  hyp={hyp[:50]}...")

    avg_wer = total_wer / total_ref_words if total_ref_words else 0.0
    avg_cer = total_cer / total_ref_chars if total_ref_chars else 0.0
    print(f"\n=== Results ===")
    print(f"Files: {len(results)}")
    print(f"CER (character-level, weighted): {avg_cer:.1%}")
    print(f"WER (word-level, weighted):      {avg_wer:.1%}")

    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump({"avg_wer": avg_wer, "avg_cer": avg_cer, "n_files": len(results), "results": results}, f, ensure_ascii=False, indent=2)
        print(f"Saved to {output_json}")


def main():
    ap = argparse.ArgumentParser(description="Benchmark live-transcribe server on SOVA dataset")
    ap.add_argument("--server", default="ws://localhost:9745", help="WebSocket server URL")
    ap.add_argument("--dataset", required=True, help="hf:name, local:path, or dir:path")
    ap.add_argument("--split", default="test", help="HuggingFace split (test/validation/train)")
    ap.add_argument("--limit", type=int, default=None, help="Max number of files")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("-o", "--output", dest="output_json", help="Save results JSON")
    args = ap.parse_args()

    asyncio.run(run_benchmark(
        server_url=args.server,
        dataset_spec=args.dataset,
        split=args.split,
        limit=args.limit,
        verbose=args.verbose,
        output_json=args.output_json,
    ))


if __name__ == "__main__":
    main()
