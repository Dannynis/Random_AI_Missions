"""Download and decode the public audio used by A1 into int16 16 kHz arrays.

Sources (Hugging Face datasets, parquet):
  speech  LibriSpeech train-clean-100 (shards 0-2) for training, test-clean for testing
  music   Free Music Archive small (shards 0-1 train, shard 14 test)
  noise   ESC-50 (folds 1-4 train, fold 5 test), used as background noise in the channel
Output dir: $A1_DATA (default $HF_HOME/a1_audio). Each split is one int16 array of
concatenated audio plus a JSON index of (start, length, group) per source file.
Also builds simulated room impulse response banks (image source method, disjoint
rooms for train / test).
"""
from __future__ import annotations

import io
import json
import math
import os
import sys

import numpy as np

SR = 16000


def out_dir() -> str:
    d = os.environ.get("A1_DATA") or os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "a1_audio")
    os.makedirs(d, exist_ok=True)
    return d


def fetch(repo: str, path: str) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, path, repo_type="dataset")


def decode(b: bytes) -> np.ndarray:
    import soundfile as sf
    from scipy.signal import resample_poly
    try:
        y, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=True)
    except Exception:
        import librosa
        y, sr = librosa.load(io.BytesIO(b), sr=None, mono=False)
        y = np.atleast_2d(y).T
    y = y.mean(1)
    if sr != SR:
        g = math.gcd(int(sr), SR)
        y = resample_poly(y, SR // g, int(sr) // g).astype(np.float32)
    return y


def build(name: str, files: list[tuple[str, str]], group_col: str | None, max_hours: float,
          max_per_group: float = 1e9, keep=lambda r: True):
    d = out_dir()
    if os.path.exists(os.path.join(d, f"{name}.json")):
        print("exists", name); return
    import pyarrow.parquet as pq
    chunks, index, total, per_g = [], [], 0, {}
    for repo, path in files:
        p = fetch(repo, path)
        cols = ["audio"] + ([group_col] if group_col else [])
        pf = pq.ParquetFile(p)
        for batch in pf.iter_batches(batch_size=64, columns=[c for c in cols if c in pf.schema_arrow.names] or ["audio"]):
            for r in batch.to_pylist():
                if not keep(r):
                    continue
                g = str(r.get(group_col, "")) if group_col else ""
                if per_g.get(g, 0) >= max_per_group * SR:
                    continue
                a = r["audio"]
                try:
                    y = decode(a["bytes"])
                except Exception as e:
                    print("skip", e); continue
                if len(y) < 2 * SR or not np.isfinite(y).all():
                    continue
                pk = np.abs(y).max()
                if pk < 1e-4:
                    continue
                y = y / pk * 0.9
                index.append((total, len(y), g)); total += len(y); per_g[g] = per_g.get(g, 0) + len(y)
                chunks.append((y * 32767).astype(np.int16))
                if total >= max_hours * 3600 * SR:
                    break
            if total >= max_hours * 3600 * SR:
                break
        if total >= max_hours * 3600 * SR:
            break
    np.save(os.path.join(d, f"{name}.npy"), np.concatenate(chunks))
    json.dump(index, open(os.path.join(d, f"{name}.json"), "w"))
    print(name, f"{total / SR / 3600:.2f} h", len(index), "files", flush=True)


def rir_banks():
    d = out_dir()
    if os.path.exists(os.path.join(d, "rir_test.npy")):
        return
    import pyroomacoustics as pra
    for name, seed, n in (("rir_train", 1, 3000), ("rir_test", 2, 400)):
        rng = np.random.default_rng(seed)
        out = np.zeros((n, 8000), np.float32)
        for i in range(n):
            while True:
                L = rng.uniform([3, 3, 2.4], [12, 10, 4.5])
                rt60 = rng.uniform(0.15, 1.0)
                try:
                    e_abs, max_order = pra.inverse_sabine(rt60, L)
                except ValueError:
                    continue
                room = pra.ShoeBox(L, fs=SR, materials=pra.Material(e_abs), max_order=min(max_order, 20))
                src = rng.uniform(0.3, L - 0.3)
                dist = rng.uniform(0.3, 4.0)
                v = rng.normal(size=3); v /= np.linalg.norm(v)
                mic = np.clip(src + dist * v, 0.2, L - 0.2)
                room.add_source(src); room.add_microphone(mic)
                room.compute_rir()
                h = np.asarray(room.rir[0][0], np.float32)
                k = int(np.argmax(np.abs(h)))
                h = h[max(0, k - 16):][:8000]          # direct path near sample 16
                out[i, :len(h)] = h / (np.linalg.norm(h) + 1e-9)
                break
        np.save(os.path.join(d, f"{name}.npy"), out)
        print(name, out.shape, flush=True)


def main():
    tiny = "--tiny" in sys.argv
    h = 0.3 if tiny else 1.0
    LS, FMA, ESC = "openslr/librispeech_asr", "benjamin-paine/free-music-archive-small", "ashraq/esc50"
    build("speech_test", [(LS, "all/test.clean/0000.parquet")], "speaker_id", 3 * h, max_per_group=240)
    build("music_test", [(FMA, "data/train-00014-of-00015.parquet")], None, 2 * h)
    build("noise_test", [(ESC, "data/train-00001-of-00002-27425e5c1846b494.parquet")], "fold", 1,
          keep=lambda r: r.get("fold", 5) == 5)
    build("noise_train", [(ESC, "data/train-00000-of-00002-2f1ab7b824ec751f.parquet"),
                          (ESC, "data/train-00001-of-00002-27425e5c1846b494.parquet")], "fold", 3,
          keep=lambda r: r.get("fold", 1) != 5)
    rir_banks()
    build("speech_train", [(LS, f"all/train.clean.100/{i:04d}.parquet") for i in range(3)], "speaker_id", 15 * h)
    build("music_train", [(FMA, f"data/train-{i:05d}-of-00015.parquet") for i in range(2)], None, 12 * h)


if __name__ == "__main__":
    main()
