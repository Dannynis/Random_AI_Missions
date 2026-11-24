# A1: over-the-air audio watermarking vs AudioSeal and WavMark (positive)

**2026-11-24** · code `experiments/a1_ota_watermark` · results `results/a1_ota_watermark/` · trained on one 48 GB GPU

## Question
Can a watermark survive being played through a loudspeaker and re-recorded by a microphone — the "over-the-air"
channel — and still let a receiver read a 16-bit message, where the published neural watermarks AudioSeal and WavMark
fail? Origin: Random-Ideas PRs #19, #33, #45, #57, which took *simulated* over-the-air message success from 0.025 to
~0.8 on small models and 2-3 seeds. This scales it to a full-size system on real speech and music, adds matched-audibility
baselines, a real loudspeaker/microphone recording, a stress set of unseen channels, and 5 seeds.

## System
A learned physical layer — a ~1 M-parameter encoder and ~2 M-parameter decoder — embeds a mark shaped by the host
audio, trained on 15 h LibriSpeech speech + 8.8 h Free Music Archive music through a **differentiable simulated channel**
(image-source room impulse responses with disjoint test rooms, loudspeaker/microphone colouring, real background
noise, resampling / clock drift, MP3). The link layer from the origin project is reused unchanged: a convolutional code
with soft-Viterbi decoding, CRC-8, and a blind joint timing/phase search so the receiver needs no synchronisation side
information. Message success is counted at a **<= 1 % false-alarm** operating point with per-source two-fold
thresholds, so a method cannot buy success by raising its false-alarm rate.

Baselines: the public **AudioSeal** (16-bit) and **WavMark** checkpoints, at **matched audibility** (PESQ 4.0 and
4.3) and at native strength. AudioSeal is given a timing grid search (its decoder needs 320-sample grid alignment,
although its detector is shift-tolerant); WavMark's own pattern search is already shift-robust.

## Pass / fail (stated before the run)
**Pass:** at PESQ 4.0 and 4.3, ours beats both AudioSeal and WavMark on `ota` and `ota_mp3` message success at
<= 1 % false alarm by at least 0.10 absolute, averaged over 5 seeds, with the seed spread not overlapping.

## Result — simulated channel, 5 seeds (16-bit message success at <= 1 % false alarm)

| method | audibility | clean | mp3 64k | opus 32k | reverb | ota | ota+mp3 |
|---|---|---|---|---|---|---|---|
| **ours** | PESQ 4.0 | 1.000 | 1.000 | 1.000 | 1.000 | **0.997 ± 0.001** | **0.995 ± 0.003** |
| AudioSeal | PESQ 4.0 | 0.697 | 0.693 | 0.719 | 0.048 | 0.037 ± 0.004 | 0.037 ± 0.003 |
| WavMark | PESQ 4.0 | 1.000 | 0.996 | 0.982 | 0.600 | 0.330 ± 0.018 | 0.223 ± 0.021 |
| **ours** | PESQ 4.3 | 1.000 | 1.000 | 1.000 | 0.998 | **0.983 ± 0.005** | **0.977 ± 0.010** |
| AudioSeal | PESQ 4.3 | 0.634 | 0.697 | 0.708 | 0.054 | 0.051 ± 0.005 | 0.037 ± 0.010 |
| WavMark | PESQ 4.3 | 1.000 | 0.967 | 0.889 | 0.347 | 0.126 ± 0.018 | 0.077 ± 0.015 |

**Verdict: pass, decisively.** On both over-the-air channels at both audibility levels, ours clears the nearest
baseline by 0.65-0.90 absolute with non-overlapping seed spreads (criterion was 0.10). WavMark degrades gracefully
through the room (0.33 / 0.13 at `ota`); AudioSeal's message decoder collapses to chance over the air even with the
grid search, though its *detector* still fires (it knows a mark is present, it just cannot read it). On clean and
codec-only channels all three are strong; the gap is specific to re-recording.

Figures: `fig_success_fa1.png` (per-channel bars, 5 seeds), `fig_training.png` (BER and LSD during training), `table.md`.

## Audibility is matched on PESQ, not on energy
At PESQ 4.0 ours sits at SI-SNR 22.3 dB against WavMark's 33.2 dB — ours carries ~11 dB more residual energy at the
same PESQ, because its mask is host-shaped (EQ-like), which PESQ partly forgives. So the comparison above is matched on
PESQ only. To rule out that the win is a PESQ blind spot, an **energy-matched** evaluation gives the baselines equal or
greater residual energy (seed 0 so far):

| audibility target | ours `ota` | AudioSeal `ota` | ours `ota+mp3` | AudioSeal `ota+mp3` |
|---|---|---|---|---|
| SI-SNR 22 dB (AudioSeal then at PESQ 4.28 > ours 4.04) | 1.000 | 0.050 | 0.997 | 0.037 |
| SI-SNR 26 dB (AudioSeal then at PESQ 4.45 > ours 4.34) | 0.980 | 0.007 | 0.970 | 0.017 |

Even when AudioSeal is made *louder* than ours in both PESQ and SI-SNR, it still cannot read the message over the air.
The win is not an artifact of the audibility metric. (WavMark energy-matched arm and seeds 1-4 are running.)

## Real loudspeaker and microphone (the headline test)
The 5-seed table is on the same simulator family the system trained on, so the real test is a physical recording.
One seed's streams were played on a laptop's built-in speakers and captured by its microphone array in one room
(100 host streams per arm, 50 speech / 50 music, 1000 items; alignment correlation median 0.40, no clipping), with
unmarked twins to measure false alarms. Message success at <= 1 % false alarm:

| method | PESQ 4.0 | PESQ 4.3 |
|---|---|---|
| **ours** (CRC, list of 4) | **1.00** | **0.94** |
| AudioSeal | 0.00 | 0.02 |
| WavMark | 0.00 | 0.00 |

Ours: no CRC false alarm on the 100 unmarked twins, bit error 0.000 at PESQ 4.0. AudioSeal still *detects* its mark
over the real air (AUC 0.999) but cannot read it (bit error 0.31); WavMark finds nothing (AUC 0.50). Re-encoding the
real recording afterwards: ours keeps 0.98 / 0.99 / 0.97 through MP3 64k / AAC 32k / telephone band at PESQ 4.0, and
breaks only under 12 kbit/s Opus (0.40) — a codec never seen in training applied on top of a real room. The baselines
were already at chance before any codec.

## Stress set — unseen channels (seed 0, preliminary; WavMark pending)
Channels absent from training, ours at PESQ 4.0 vs AudioSeal:

| channel | ours | AudioSeal |
|---|---|---|
| background noise 0-5 dB SNR (`ota_noisy`) | 0.933 | 0.027 |
| 1 % playback-speed / clock error (`speed_1pct`) | 1.000 | 0.000 |
| telephone band + mu-law (`phone`) | 1.000 | 0.627 |
| AAC 32k | 1.000 | 0.583 |
| Opus 12k | 0.983 | 0.557 |

The blind timing search handles a 1 % clock error that puts AudioSeal at zero. Note Opus 12k *alone* (in simulation)
leaves ours at 0.98, while Opus 12k *after a real room* dropped it to 0.40: the two distortions compound. Seed 0 only;
full 5-seed stress and the WavMark column are running.

## Training-channel ablation (2 seeds)
The same model trained on the origin's *simple* channel instead of the realistic one still reaches 0.985 on simulated
`ota` at PESQ 4.0 (vs 0.997 for the realistic-channel model) — on the in-distribution test the realistic channel adds
only ~0.01. Its value should show on the stress set and the real recording, where the simple-channel model has not yet
been run; that comparison is the honest way to justify the channel and is the next ablation to finish.

## Prior work
Learned over-the-air watermarking is not itself new. **DeAR** (AAAI 2023, arXiv 2212.02339) trains an encoder/decoder
through a re-recording distortion layer and reports ~98.5 % bit accuracy at 20 cm on real re-recordings. The SoK of
arXiv 2503.19176 already shows that current watermarks, AudioSeal included, fail physical-level distortion. What A1
adds over that literature: **blind** decoding with message-level success at a fixed false-alarm rate (not bit accuracy
with known timing), a **matched-audibility** comparison against the modern baselines including an energy-matched
control, and a **real-room** recording. A DeAR baseline (reimplemented from the paper, no public code) is training now
for a like-for-like comparison.

## Verdict and limitations
**The pre-stated criterion is met with a wide margin on 5 seeds, and the result holds on a real loudspeaker/microphone
recording and under an energy-matched control.** AudioSeal and WavMark, at matched audibility, cannot read a message
over the air; ours reads it at ~0.98-1.00.

Honest limits before this is a paper: (1) real recording is one room, one device pair, one distance, seed 0 only —
needs several rooms / distances / devices and more seeds; (2) the audibility match needs a second metric (ViSQOL) and
a formal ABX listening test — the one informal blind check (10 sets, one listener) did not separate the methods;
(3) DeAR must be added as a baseline (in progress); (4) the simple-channel ablation must be run on the stress and real
sets; (5) low-rate Opus after a real room is a genuine failure mode to characterise, not hide.
