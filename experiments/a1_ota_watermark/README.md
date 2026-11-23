# A1: over-the-air audio watermarking at scale, against AudioSeal and WavMark

Backlog item A1. Origin: Random-Ideas PRs #19 (physical layer), #33 (soft-decision convolutional code),
#45 (blind synchronisation), #57 (CRC lock rejection). On small CPU models and 2-3 seeds that stack took
simulated over-the-air 16-bit message success (8 s recordings, random start) from 0.025 to about 0.70-0.73
at <= 1 % false alarm.

## Hypothesis

A learned, host-shaped STFT-mask watermark trained end to end through a broad play-and-record channel, read
by the original link layer (rate-1/2 K=7 convolutional code over a 4-chunk cycle, CRC-8, list Viterbi, joint
ML timing search over the decoder's sliding window), delivers more correct 16-bit messages from 8 s
re-recorded audio than the public AudioSeal and WavMark checkpoints **at the same audibility**.

## What changed from the small-scale runs

| | Small runs (origin) | This experiment |
|---|---|---|
| Hosts | LibriSpeech test-clean + ESC-50, 2 s crops | LibriSpeech train-clean-100 (15 h) + Free Music Archive (8.8 h) for training; LibriSpeech test-clean + held-out FMA shard for test |
| Model | 16-channel 3-layer encoder, 32-channel decoder | 64-channel 6-block dilated residual encoder with payload injected at every block; 4-stage residual decoder (64-128 channels) |
| Training | 3000 steps, batch 16, isolated 2 s clips | 40k steps, batch 32 three-chunk streams, decoder trained on windows with timing jitter at real chunk boundaries |
| Channel | exponential-noise RIRs, band-pass, AWGN, mu-law | image-source room impulse responses (3000 train rooms, 400 disjoint test rooms, 0.3-4 m, RT60 0.15-1.0 s), loudspeaker response + saturation, microphone response, real background sounds (ESC-50, disjoint folds), white / pink sensor noise, clock drift, plus the original channel |
| Baselines | classical spread spectrum | AudioSeal (16-bit) and WavMark (16-bit payload + 16 pattern bits), public checkpoints |
| Seeds | 2-3 | 5 model seeds, each on its own test streams; baselines run on the same streams (paired) |

The link layer in `legacy/` is copied unchanged from the origin branch (`wm_coding.py`, `wm_sync.py`,
`wm_crc.py`, and `audio_watermark.py` for its constants). The new physical layer keeps the same interface
(2 s chunks, 32 slots, hop 256, conv trunk -> window mean -> head) so the receiver runs on it unchanged.

## Protocol

- **Payload:** 16 information bits per recording for every method. Ours appends CRC-8 and codes over 4 x 2 s
  chunks looped through the stream; AudioSeal carries the 16 bits directly; WavMark carries them as its 16-bit
  payload in every 1 s segment.
- **Streams:** 12 s marked host streams (half speech, half music), each recording an 8 s window starting at a
  random point inside (unknown timing and cycle phase). 300 recordings per seed.
- **Matched audibility:** each method's watermark residual is rescaled so that the mean wideband PESQ over a
  separate calibration set of speech streams is 4.0, and again 4.3. The native strength of each method is also
  reported. SI-SNR and log-spectral distance are logged at every operating point.
- **Channels (test):** `clean`, `mp3_64k`, `opus_32k`, `reverb` (held-out rooms), `ota_legacy` (the origin's
  over-the-air recipe), `ota` (loudspeaker response + saturation, held-out room, microphone response, background
  sound at 15-25 dB SNR, sensor noise, +-100 ppm clock drift), `ota_mp3` (`ota` then MP3 64 kbit/s). Codecs are
  never seen in training.
- **Metrics:** message success at <= 1 % false alarm (thresholds set on unmarked twins that went through the
  identical channel draw, two-fold), raw success and false alarm with no threshold, message bit error rate,
  detection AUC, split by speech / music. Ours uses `crc@1` (no threshold) as the primary receiver.

## Pass / fail (stated before the run)

**Pass:** at PESQ 4.0 and at 4.3, ours beats both AudioSeal and WavMark on `ota` and `ota_mp3` message success at
<= 1 % false alarm by at least 0.10 absolute, averaged over 5 seeds, with the seed spread not overlapping.
**Fail:** either baseline is within 0.10 or better on either over-the-air channel at either audibility.
Results on `clean` / codec-only channels are reported but are not part of the criterion.

## Prior work checked

- AudioSeal (San Roman et al., 2024) and WavMark (Chen et al., 2023): the two public baselines used here.
- AudioMarkBench (arXiv 2406.06979, 2024) and the Sony robustness study (arXiv 2505.19663, 2025) test these models
  against noise, filtering, reverberation, codecs and EQ, but not a full loudspeaker -> room -> microphone chain.
- SoK on audio watermark robustness (arXiv 2503.19176, 2025) lists play-and-re-record as a known threat.
  No public comparison of channel-coded, blindly synchronised learned watermarks against these baselines
  under a simulated acoustic chain was found.

## Caveats known in advance

- The channel is simulated; a real loudspeaker / microphone recording set is the next step if this passes.
- PESQ is a speech metric; music audibility is reported through SI-SNR and LSD only.
- The baselines were trained for 1 s localisation (AudioSeal) and segment-wise decoding (WavMark), not for
  8 s messages; both are used through their own public decoders with no retraining.

## Run

`run.sh tiny` (validation, a few minutes) or `run.sh` (5 seeds; `SEEDS` / `STEPS` override). Everything resumes
from `$RUN_DIR/ckpt` and skips finished evaluation blocks. Audio is prepared by `prep_data.py` into `$A1_DATA`
(default under `$HF_HOME`).
