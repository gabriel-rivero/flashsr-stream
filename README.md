# flashsr-stream

Real-time 48 kHz audio upscaling from the terminal, using
[FlashSR](https://github.com/jakeoneijk/FlashSR_Inference) (a one-step distilled
[AudioSR](https://github.com/haoheliu/versatile_audio_super_resolution)).

Three ways to use it:

| command | what it does |
|---|---|
| `audiosr in.mp3 out.flac` | batch upscale a file (windowed, crossfaded) |
| `audiosr-play track.mp3` | upscale a file and play it live on a PulseAudio/PipeWire sink — no intermediate file |
| `audiosr-sink` | create a virtual sink, upscale everything routed into it, play it on the real device |

FlashSR is a *one-step* diffusion model, so a 5.12 s window costs a single UNet
forward plus a vocoder pass. On an RTX 4060 Ti that is **0.96 s for mono and
1.89 s for stereo** (bf16), i.e. RTF 0.19 / 0.37 — comfortably faster than
realtime. That is what makes the streaming modes possible at all; the original
AudioSR needs 50 diffusion steps and cannot keep up.

---

## Install

### 1. Dependencies

Python 3.10–3.12 and a CUDA-capable PyTorch build:

```bash
git clone <this repo> flashsr-stream
cd flashsr-stream

python -m venv venv                       # or: uv venv
./venv/bin/pip install -r requirements.txt
```

`requirements.txt` pulls the CPU build of torch by default. For CUDA, install
torch first from the index that matches your driver, then the rest:

```bash
./venv/bin/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
./venv/bin/pip install -r requirements.txt
```

### 2. The FlashSR package

Either install it (recommended):

```bash
./venv/bin/pip install git+https://github.com/jakeoneijk/FlashSR_Inference
```

or clone it next to the script:

```bash
git clone https://github.com/jakeoneijk/FlashSR_Inference flashsr
```

The script looks for the package in `$FLASHSR_DIR`, `./flashsr`,
`./vendor/flashsr`, an installed `FlashSR` module, and finally the usual ComfyUI
`custom_nodes` locations — so it also works unchanged inside an existing
[ComfyUI-AudioSR](https://github.com/gabriel-rivero/ComfyUI-AudioSR) install.

### 3. Weights (~3.3 GB)

```bash
./audiosr --download
```

That fetches `student_ldm.pth`, `sr_vocoder.pth` and `vae.pth` from
`jakeoneijk/FlashSR_weights` into `./models` (or `$FLASHSR_MODEL_DIR`, or the
ComfyUI models directory if the weights are already there).

### 4. Verify and put it on $PATH

```bash
./audiosr --check        # preflight: package, deps, weights, GPU
./install.sh             # symlinks audiosr* into ~/.local/bin
```

`ffmpeg` is required for the streaming modes, and `pacat`/`parec`/`pactl`
(`pulseaudio-utils`) for the sink modes.

---

## Usage

```bash
# batch: writes in.flac next to the input
audiosr in.mp3

# batch with explicit output and overlap
audiosr in.mp3 out.flac --overlap 1.5

# stream a file to the default output device
audiosr-play track.mp3

# lower latency (2.56 s window), pick a device
audiosr-play track.mp3 --window 2.56 --sink alsa_output.pci-0000_0d_00.4.iec958-stereo

# same pipeline, but write FLAC instead of playing
audiosr-play track.mp3 --dump out.flac

# system-wide middleware sink
audiosr-sink                       # creates audiosr_in, plays on the real default
audiosr-sink --take-default        # ...and makes it the default while running
pactl move-sink-input <id> audiosr_in
```

`audiosr-play --list-sinks` prints the available sinks.

### Useful flags

| flag | meaning |
|---|---|
| `--window 2.56` | FlashSR window in seconds; must be a multiple of 0.64. 5.12 is the trained length, smaller = lower latency, less context |
| `--overlap 1.0` | crossfade between windows |
| `--dtype bf16` | bf16 autocast (default on CUDA; 1.2× faster and 1.5 GB lighter than fp32) |
| `--cutoff 16000` | fix the lowpass cutoff instead of auto-detecting it once |
| `--no-lowpass` | feed the model the unmodified signal |
| `--skip-silence` | pass digital silence through without a GPU pass (keeps an idle daemon at ~1% GPU) |
| `--device cpu` | run on CPU (works, but see the numbers below) |

---

## How the streaming works

```
file ──ffmpeg──┐
               ├─> ring buffer ─> 5.12 s window ─> FlashSR (bf16) ─> RMS match ─> cosine crossfade ─> pacat ─> sink
sink.monitor ──┘                 (advance by window − overlap)      (gain clamp)     (overlap)
```

* The model only accepts windows that are a multiple of **30720 samples
  (0.64 s)**, minimum one granule.
* Each window holds back its last `overlap` samples and blends them with the
  next window's head, so the emitted sample count matches the input exactly and
  there are no seams.
* Output level is glued to input level with a gain clamped to 0.2–5×, seeded
  from the first window and then smoothed (EMA) so windows do not pump.
* Writes go through a writer thread in 0.25 s chunks. Writing to a realtime sink
  blocks; doing that on the inference thread serializes playback with compute and
  drops the stream below realtime (RTF 1.47 instead of 0.37).
* `--listen` resolves and pins `@DEFAULT_SINK@` *before* `--take-default`
  switches the default, so the daemon can never feed itself.

Latency is **window + inference time**: about 6–7 s at the default 5.12 s
window, ~2.8 s at 1.28 s. This is music/podcast tooling — it cannot track video
lip-sync.

---

## Performance

Measured on an RTX 4060 Ti 8 GB, Ryzen 9 5900XT, torch 2.15 + CUDA 13.2,
`num_steps=1`, bf16 autocast:

| window | channels | time / window | RTF | peak VRAM |
|---|---|---|---|---|
| 5.12 s | mono | 0.96 s | 0.19 | 3.2 GB |
| 5.12 s | stereo | 1.89 s | 0.37 | 3.7 GB |
| 2.56 s | stereo | 0.96 s | 0.38 | 3.2 GB |
| 0.64 s | stereo | 0.25 s | 0.39 | 2.9 GB |
| 5.12 s | stereo (fp32) | 2.31 s | 0.45 | 5.2 GB |

`torch.compile` gives only a few percent on this model (1–3% for `default`,
5.6% for `reduce-overhead` at the cost of a ~90 s startup), because it is
bandwidth-bound and already runs on fused SDPA kernels. Details in
[`docs/measurements.md`](docs/measurements.md).

## Portability

**Minimum install:** Python 3.10+, `torch` (CUDA build), `numpy`, `scipy`,
`librosa`, `soundfile`, `einops`, `pyyaml`, `matplotlib`, `tqdm`,
`huggingface_hub`, plus the FlashSR inference package and its 3.3 GB of weights.
That set was verified in a clean venv: `transformers`, `timm`, `torchaudio` and
`safetensors` are **not** needed for the FlashSR path (they are only pulled in by
the optional `--engine audiosr` fallback). No ComfyUI, no diffusers, no
`xformers`/`sageattention`. `ffmpeg` for the streaming modes;
`pulseaudio-utils` for the sink modes.

**Minimum hardware:** any CUDA GPU with ~4 GB free VRAM for stereo bf16 (2.9 GB
peak at a 0.64 s window, 3.7 GB at 5.12 s). For *realtime* streaming you need
roughly RTF < 0.8 to leave headroom: an RTX 3060-class card (~2× slower than a
4060 Ti) lands around RTF 0.7 and is the practical floor. Anything older/slower
will still do batch upscaling, just not live playback.

**OS:** the batch mode is pure Python + ffmpeg and runs anywhere torch does.
`--play`/`--listen` are Linux-only because they drive PulseAudio/PipeWire
directly (`pacat`, `parec`, `pactl`). macOS would need a CoreAudio/BlackHole
equivalent, Windows a WASAPI loopback equivalent; the DSP path itself is portable.

CPU-only works but is 2–9× slower than realtime (0.64 s mono window: RTF 2.0;
5.12 s stereo: RTF 8.7, ~6 GB RAM), so batch only. Full tables in
[`docs/measurements.md`](docs/measurements.md).

---

## Credits and licence

* [FlashSR](https://arxiv.org/abs/2501.10807) by Jaekwon Im et al. —
  inference code (`jakeoneijk/FlashSR_Inference`) and weights
  (`jakeoneijk/FlashSR_weights`).
* [AudioSR](https://github.com/haoheliu/versatile_audio_super_resolution) by
  Haohe Liu et al. — the model FlashSR distills.
* This wrapper: MIT (see `LICENSE`). The weights are **not** included and keep
  their upstream licence; check the FlashSR/AudioSR repositories before
  redistributing them.
