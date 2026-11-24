"""Recording side of the A1 real-world check (needs only numpy, scipy, sounddevice).

usage: python real_play.py <dir holding playlist.wav> [--api ...] [--in-name ...] [--out-name ...] [--rate 48000]
Plays playlist.wav through a loudspeaker while recording a microphone on one full-duplex stream (one clock), at the
rate given (default: the system mixer at 16 kHz mono; `--api "Windows WDM-KS" --rate 48000 --channels 2` uses kernel
streaming instead: raw, no system volume or voice processing). Do not combine a fixed block size with high latency on
the system mixer path: on some drivers that hangs the stream open.
`--in-api "Windows WDM-KS" --in-rate 48000 --in-channels 2` records on a separate input stream (own clock) while the
playlist plays on --api; the scorer re-aligns every item, so the small drift between the two clocks does not matter.
Writes recording.wav (16 kHz mono, first mic channel) next to the playlist; the raw device-rate capture is kept as
recording_raw.wav.
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
import scipy.signal as ss
import sounddevice as sd


def keep_awake(on=True):
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | (0x00000001 if on else 0))


def find(api, name, kind):
    for i, d in enumerate(sd.query_devices()):
        if (sd.query_hostapis(d["hostapi"])["name"] == api and name.lower() in d["name"].lower()
                and d[f"max_{kind}_channels"] > 0):
            return i, d
    raise SystemExit(f"no {kind} device matching {name!r} on {api!r}")


def resample(x, up, down, chunk=1 << 20, pad=4096):
    """Polyphase resampling in overlapping chunks (bounded memory for multi-hour signals)."""
    out = []
    for s in range(0, len(x), chunk):
        a, b = max(0, s - pad), min(len(x), s + chunk + pad)
        y = ss.resample_poly(x[a:b].astype(np.float32), up, down)
        lo = (s - a) * up // down
        out.append(y[lo: lo + (min(len(x), s + chunk) - s) * up // down])
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--api", default="MME", help='host API; "Windows WDM-KS" bypasses the system mixer (48000 Hz x2)')
    ap.add_argument("--in-name", default="Microphone Array")
    ap.add_argument("--out-name", default="Speakers")
    ap.add_argument("--rate", type=int, default=16000)
    ap.add_argument("--channels", type=int, default=1)
    ap.add_argument("--in-api", default=None, help="record on a separate input stream on this host API")
    ap.add_argument("--in-rate", type=int, default=48000)
    ap.add_argument("--in-channels", type=int, default=2)
    ap.add_argument("--out", default="recording.wav")
    a = ap.parse_args()
    sr0, pl = wf.read(os.path.join(a.dir, "playlist.wav"))
    assert pl.dtype == np.int16 and pl.ndim == 1
    g = np.gcd(a.rate, sr0)
    pl = np.clip(resample(pl.astype(np.float32) / 32767, a.rate // g, sr0 // g), -1, 1).astype(np.float32)
    total = len(pl)
    split = a.in_api is not None
    rate_in = a.in_rate if split else a.rate
    ii, di = find(a.in_api if split else a.api, a.in_name, "input"); oo, do = find(a.api, a.out_name, "output")
    print(f"in [{ii}] {di['name']} | out [{oo}] {do['name']} | {a.api} {a.rate} Hz x{a.channels}"
          f" | {total / a.rate / 60:.1f} min", flush=True)
    q: queue.Queue = queue.Queue()
    pos = [0]
    status_log = []

    def cb(indata, outdata, frames, t, status):
        if status:
            status_log.append(str(status))
        p = pos[0]
        chunk = pl[p:p + frames]
        outdata[:len(chunk)] = chunk[:, None]
        outdata[len(chunk):] = 0
        pos[0] = p + frames
        if indata is not None:
            q.put((np.clip(indata[:, 0], -1, 1) * 32767).astype(np.int16))
        if p >= total + 2 * a.rate:                       # 2 s of room tail after the playlist
            raise sd.CallbackStop

    raw = os.path.join(a.dir, "recording_raw.wav")
    w = wave.open(raw, "wb")
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate_in)
    done = [False]

    def cb_out(outdata, frames, t, status):
        cb(None, outdata, frames, t, status)

    def cb_in(indata, frames, t, status):
        if status:
            status_log.append("in " + str(status))
        if not done[0]:
            q.put((np.clip(indata[:, 0], -1, 1) * 32767).astype(np.int16))
    keep_awake(True)
    t0 = time.time(); last = 0; peak = 0; written = 0
    try:
        import contextlib
        with contextlib.ExitStack() as es:
            if split:
                es.enter_context(sd.InputStream(samplerate=rate_in, channels=a.in_channels, dtype="float32", device=ii,
                                                callback=cb_in))
                s = es.enter_context(sd.OutputStream(samplerate=a.rate, channels=a.channels, dtype="float32",
                                                     device=oo, callback=cb_out))
            else:
                s = es.enter_context(sd.Stream(samplerate=a.rate, channels=a.channels, dtype="float32",
                                               device=(ii, oo), callback=cb))
            while s.active or (not split and not q.empty()):
                try:
                    blk = q.get(timeout=0.5)
                except queue.Empty:
                    continue
                w.writeframes(blk.tobytes()); written += len(blk)
                peak = max(peak, int(np.abs(blk.astype(np.int32)).max()))
                if time.time() - last > 30:
                    last = time.time()
                    print(f"{pos[0] / a.rate / 60:6.1f} / {total / a.rate / 60:.1f} min  mic peak {peak / 32767:.3f}"
                          f"  status events {len(status_log)}", flush=True)
                    peak = 0
            done[0] = True                              # split mode: stop taking input, then drain
            while not q.empty():
                blk = q.get(); w.writeframes(blk.tobytes()); written += len(blk)
    finally:
        w.close(); keep_awake(False)
    print(f"played in {(time.time() - t0) / 60:.1f} min, {written / rate_in:.0f} s recorded, status events {len(status_log)}",
          flush=True)
    if status_log:
        print("status:", sorted(set(status_log))[:10])
    if written / rate_in < 0.99 * total / a.rate:
        raise SystemExit(f"stream stopped early: {written / rate_in:.0f} s of {total / a.rate:.0f} s")
    _, r = wf.read(raw)
    gi = np.gcd(rate_in, sr0)
    y = resample(r.astype(np.float32) / 32767, sr0 // gi, rate_in // gi)
    wf.write(os.path.join(a.dir, a.out), sr0, (np.clip(y, -1, 1) * 32767).astype(np.int16))
    print("wrote", a.out, f"{len(y) / sr0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
