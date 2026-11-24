"""Recording side of the A1 real-world check (needs only numpy, scipy, sounddevice).

usage: python real_play.py <dir holding playlist.wav> [--device-in N] [--device-out N] [--start-s S]
Plays playlist.wav through the default loudspeaker while recording the default microphone (16 kHz mono,
full duplex on one stream so both share a clock), and writes recording.wav next to it as it goes.
"""
from __future__ import annotations

import argparse
import os
import queue
import sys
import time
import wave

import numpy as np
import scipy.io.wavfile as wf
import sounddevice as sd


def keep_awake(on=True):
    if sys.platform == "win32":
        import ctypes
        # ES_CONTINUOUS | ES_SYSTEM_REQUIRED (| ES_DISPLAY_REQUIRED not needed)
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | (0x00000001 if on else 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--device-in", type=int, default=None)
    ap.add_argument("--device-out", type=int, default=None)
    ap.add_argument("--out", default="recording.wav")
    a = ap.parse_args()
    sr, pl = wf.read(os.path.join(a.dir, "playlist.wav"))
    assert pl.dtype == np.int16 and pl.ndim == 1
    total = len(pl)
    q: queue.Queue = queue.Queue()
    pos = [0]
    status_log = []

    def cb(indata, outdata, frames, t, status):
        if status:
            status_log.append(str(status))
        p = pos[0]
        chunk = pl[p:p + frames]
        outdata[:len(chunk), 0] = chunk
        outdata[len(chunk):, 0] = 0
        pos[0] = p + frames
        q.put(indata[:, 0].copy())
        if p >= total:
            raise sd.CallbackStop

    w = wave.open(os.path.join(a.dir, a.out), "wb")
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
    keep_awake(True)
    t0 = time.time(); last = 0; peak = 0; written = 0
    try:
        with sd.Stream(samplerate=sr, channels=1, dtype="int16", blocksize=1024, latency="high",
                       device=(a.device_in, a.device_out), callback=cb) as s:
            while s.active or not q.empty():
                try:
                    blk = q.get(timeout=0.5)
                except queue.Empty:
                    continue
                w.writeframes(blk.tobytes()); written += len(blk)
                peak = max(peak, int(np.abs(blk.astype(np.int32)).max()))
                if time.time() - last > 30:
                    last = time.time()
                    print(f"{pos[0] / sr / 60:6.1f} / {total / sr / 60:.1f} min  mic peak {peak / 32767:.2f}"
                          f"  xruns {len(status_log)}", flush=True)
                    peak = 0
    finally:
        w.close(); keep_awake(False)
    print(f"done in {(time.time() - t0) / 60:.1f} min, {written / sr:.0f} s recorded, status events {len(status_log)}",
          flush=True)
    if status_log:
        print("status:", sorted(set(status_log))[:10])


if __name__ == "__main__":
    main()
