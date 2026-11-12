# Measurements

One machine unless noted:

* GPU: NVIDIA RTX 4060 Ti, 8 GB
* CPU: AMD Ryzen 9 5900XT (16C/32T), 39 GB RAM
* torch 2.15.0+cu132 (CUDA 13.2) / Python 3.12.16
* FlashSR `num_steps=1`, `lowpass_input=False`, bf16 autocast
* default window: 245760 samples = 5.12 s @ 48 kHz (the trained length)
* RTF = seconds of compute per second of audio (lower is faster; < 1 is realtime)

## Throughput vs window size (eager, stereo)

The UNet is fully convolutional, so cost scales linearly with the window: RTF
is flat and a smaller window buys latency at no throughput cost. Valid windows
are multiples of 30720 samples (0.64 s).

| window | samples | time / window | RTF | peak VRAM |
|---|---|---|---|---|
| 0.64 s | 30720 | 252 ms | 0.39 | 2.86 GB |
| 1.28 s | 61440 | 487 ms | 0.38 | 2.98 GB |
| 2.56 s | 122880 | 963 ms | 0.38 | 3.22 GB |
| 5.12 s | 245760 | 1889 ms | 0.37 | 3.69 GB |

Mono at 5.12 s is 962 ms (RTF 0.19) and 3.22 GB.

Precision, stereo 5.12 s:

| dtype | time | RTF | peak VRAM |
|---|---|---|---|
| bf16 autocast | 1889 ms | 0.37 | 3.69 GB |
| fp32 | 2314 ms | 0.45 | 5.23 GB |

bf16 is 1.22× faster and saves 1.5 GB, so it is the default. fp32 fits an 8 GB
card but leaves little room for the desktop compositor.

## torch.compile

Compiled the UNet (`model.model`) — the only tracable part; the surrounding
sampler has data-dependent control flow. `dynamic=False`, so each new window
size triggers its own ~80–95 s compile.

| mode | first call | mono 5.12 s | stereo 5.12 s | stereo 2.56 s |
|---|---|---|---|---|
| eager | — | 962 ms (RTF 0.19) | 1891 ms (RTF 0.37) | 963 ms (RTF 0.38) |
| `default` | 96 s | 921 ms (RTF 0.18) | 1855 ms (RTF 0.36) | 950 ms (RTF 0.37) |
| `reduce-overhead` | 86 s | 911 ms (RTF 0.18) | 1838 ms (RTF 0.36) | 909 ms (RTF 0.36) |

**Verdict: not worth it.** 1–3% on the default mode, up to 5.6% on
`reduce-overhead` (CUDA graphs), for ~1.5 min of startup. The model is
bandwidth-bound at bf16 and already runs on fused SDPA kernels, so Inductor has
little left to fuse. `max-autotune` is skipped: Inductor already reports "not
enough SMs to use max_autotune_gemm mode" on this 8 GB card, and a GEMM
autotune pass would add minutes for the same reason.

## CPU-only

Clean venv with CPU-only torch, 16 threads, no GPU in the process. The published
checkpoints carry CUDA device tags, so `torch.load` needs `map_location="cpu"`
(the script does this automatically for `--device cpu`).

| channels | window | time | RTF | peak RSS |
|---|---|---|---|---|
| 1 | 0.64 s | 1.3 s | 2.0 | 5.9 GB |
| 1 | 1.28 s | 3.2 s | 2.5 | 5.9 GB |
| 1 | 2.56 s | 8.8 s | 3.4 | 5.9 GB |
| 1 | 5.12 s | 22.1 s | 4.3 | 5.9 GB |
| 2 | 0.64 s | 3.7 s | 5.8 | 5.9 GB |
| 2 | 1.28 s | 9.1 s | 7.1 | 5.9 GB |
| 2 | 2.56 s | 22.8 s | 8.9 | 6.7 GB |
| 2 | 5.12 s | 44.6 s | 8.7 | 6.7 GB |

CPU cost grows faster than the window (attention is quadratic in sequence
length), so unlike the GPU the *smallest* window is the most efficient: RTF 2.0
mono / 5.8 stereo at 0.64 s versus 4.3 / 8.7 at 5.12 s. Still 2–9× slower than
realtime, so CPU is batch mode only (a 3-minute track takes ~25–45 minutes).

## Realtime headroom

Streaming needs RTF < ~0.8 to survive scheduling jitter and sink buffering.
Stereo at 5.12 s uses 0.37 of a 4060 Ti, so a card roughly 2× slower still
works — that puts the practical floor around an RTX 3060 / 8 GB. Cards with
4 GB can still stream if the window is kept at 0.64–1.28 s (2.86–2.98 GB peak).
Below that, batch mode only.
