# LeapSVC

[![CI](https://github.com/ayutaz/LeapSVC/actions/workflows/ci.yml/badge.svg)](https://github.com/ayutaz/LeapSVC/actions/workflows/ci.yml)

**日本語**: [README.md](README.md)

LeapSVC is an acoustic model for **singing voice conversion (SVC)**. It takes a sung WAV and
produces a mel that re-sings it in the **target singer's voice** (mel to waveform is handled by
the NHVSing vocoder). It reuses the excitation (harmonic + noise) and the rectified flow of the
singing-synthesis model [LeapSinger](https://github.com/wavtechyukky/LeapSinger) unchanged, and
**swaps only the conditioning** from "phonemes + durations" to "content features + F0".

    SVS: phonemes + durations + F0                  -> LeapSinger -> mel + F0 -> NHVSing -> WAV
    SVC: source WAV -> content / F0 / UV / loudness -> LeapSVC    -> mel + F0 -> NHVSing -> WAV

> **This repository started from [LeapSinger](https://github.com/wavtechyukky/LeapSinger) and
> is now developed independently** (trunk is `main`).
> **The existing SVS path is unchanged** (see "SVS path" near the end — same content as upstream).
> **No trained SVC weights are distributed**, because the training material's licensing is
> unresolved ([License](#license)).

## What LeapSVC is

SVS and SVC **diverge only at the condition encoder**. Excitation, flow backbone, losses, GAN,
checkpoint format, and the NHVSing-compatible mel are all shared.

| | SVS (LeapSinger) | SVC (LeapSVC) |
|---|---|---|
| Input | phonemes + durations + F0 | content + F0 + V/UV + loudness |
| Encoder | `PhonemeEncoder` + `LengthRegulator` | `ContentAdapter` |
| Model | `HarmonicAcousticModel(MultiSpk)` | `HarmonicSVCModel` (`model.arch: svc`) |
| Features | `shard.npz` (needs phoneme labels) | `svc_shard.npz` (**built from WAV alone**) |

- **The content encoder is ContentVec** — the shard stores a fixed random **256-dim** subset of
  the 768-dim layer-12 features. The raw 768 dims stay in the first-stage cache, so ablating the
  subset only needs the second stage re-run.
- **F0 comes from RMVPE.** SSL frames (50 Hz) are aligned to the mel grid (172.265625 Hz) by
  **left (hold-previous)**; the ratio is not an integer, so the method itself is part of the
  data contract.
- **F0 is supplied directly as a condition**, which is why F0 tracking, V/UV and timing come out
  cleanly (see the comparison below).
- The flow's starting point `x0` is a pseudo-mel derived from F0, so **the flow never has to draw
  the periodic structure from scratch**.

## Where this stands

**Verified.** A multi-singer base over 23 speakers / ~18 hours was pretrained for 60,000 steps
and then fine-tuned (with GAN) to Namine Ritsu. **Japanese material covers 7 speakers /
~21.9 hours.**

**How to read the numbers:** intelligibility, signal quality and brightness are only ever read
as a **gap from the ceiling**. The ceiling is the **ground-truth mel pushed back through the same
vocoder** (`*_vocoder_only.wav`, emitted by `--self-check`) — the best this vocoder can do.
**We never report absolute values as quality.**

| Tier (all Japanese, same condition) | Intelligibility (CER gap from ceiling, median of diffs) | Speaker-similarity recovery | Brightness (vs ceiling) |
|---|---:|---:|---:|
| Songs seen in training (diagnostic) | +1.0 pts | — | — |
| Unseen songs, self-reconstruction | +10.7 pts | 91.4% | 1.02x |
| Unseen songs, seen speaker | +8.9 pts | 90.8% | 0.95x |
| **Unseen speakers** (2 speakers absent from the base, n=20) | **+7.3 pts** | **90.4%** | **0.93x** |

**Unseen speakers match or beat seen ones** (2026-09-17, with loudness matched).

### Against Seed-VC (26 clips, same test set)

| Metric | LeapSVC | Seed-VC | Verdict |
|---|---:|---:|---|
| Speaker similarity (ECAPA-TDNN, 20 s) | 0.5899 | 0.5912 | essentially equal (0.2% relative) |
| F0 correlation | 0.9994 | — | LeapSVC ahead |
| V/UV agreement | 99.0% | — | LeapSVC ahead |
| Timing (onset match rate) | 73.7% | 70.8% | LeapSVC ahead |
| Blind preference (informal, N=1, 26 pairs) | 4 | **21** | **Seed-VC ahead** (1 tie) |

**We cannot write "better than Seed-VC."** We lost the blind preference, and that rule was
registered before measuring. **The gap is concentrated on unseen sources** (2 vs 18 across
20 pairs); on the target's own held-out songs the two are level (2 vs 3 with 1 tie across 6 pairs).

**We measured why we lost.** Seed-VC was brighter than LeapSVC on all 26 clips (median spectral
centroid ratio 1.80), and a brighter sound reads as louder and closer at the same RMS. But
**the target's own recordings sit at 1160 Hz and LeapSVC at 1067 Hz** — nearly identical — while
Seed-VC sits at 2149 Hz. **Brightness buys preference at the cost of resembling the target**, so
we do not add it in the model.

**Only metrics anchored to the source or the target recording can be compared across systems**
(speaker similarity, timing, F0, V/UV). **CER, signal quality and brightness cannot**, because
each system's ceiling depends on its own vocoder.

### What is not measured

- **Audio quality itself.** Content preservation, F0 tracking, V/UV, brightness and signal
  quality are measured; quality is not. Signal quality (SQUIM) is **trained on speech**, so its
  validity for singing is unverified.
- **Male and low-voiced unseen speakers.** Both unseen speakers used for evaluation are
  **female and in nearly the same range as the target**, and **time below C3 is almost absent
  from every corpus we have**.
- **Real time.** `realtime_capable` only checks `rtf_total < 1`; chunk boundaries, audio I/O and
  sustained operation are unmeasured. **We do not write "real-time."**
- **Subjective speaker similarity.** Objectively the two systems are equal (0.2% relative), and
  the listening test reached no verdict (all 4 votes landed on whichever side played last).

### RTF (measured, 20-second phrase)

| Stage | CPU | GPU |
|---|---:|---:|
| Acoustic (flow, 16 steps) | 0.081 | 0.006 |
| **Vocoder (NHVSing / ONNX, runs on CPU)** | 0.355 | **0.432** |
| Total | 0.654 | 0.464 |

**The vocoder is the largest term — 93% of the total on GPU.** Speeding up the acoustic model
barely moves end-to-end. **"1-step" refers to the acoustic flow**, not the pipeline
(**the SVC default is 16 steps**; see the table below).

## Usage (SVC)

### Setup

Python is pinned to **3.13**, and dependency management and execution go through
[uv](https://docs.astral.sh/uv/).

    git clone https://github.com/ayutaz/LeapSVC
    cd LeapSVC
    uv sync --extra train --extra export --extra dev              # develop / train / export
    uv sync --extra train --extra export --extra dev --extra eval # add the evaluation stack

**`uv sync --extra <name>` means "only these"**, not "add this" — list every extra you need
every time. Run Python with `uv run python ...` and add dependencies with `uv add <package>`
(never bare `python` / `pip`, never `uv pip`). For PyTorch, RMVPE and the vocoder, see the
setup notes in the SVS section below (they are shared).

### Preprocessing (WAV to features)

**No phoneme labels are needed** — shards are built straight from a directory of WAVs.
**Keep one directory per speaker**: the speaker id is looked up from the directory name, so
speakers mixed into one shard cannot be told apart.

    uv run python -m preprocess.svc.run --wav-dir download/ritsu --out data/ritsu_svc

    # deeply nested corpora (<technique>/<song>/<Group>/NNNN.wav) need --song-parts
    uv run python -m preprocess.svc.run --wav-dir download/gtsinger/Japanese/JA-Soprano-1 \
      --out data/JA_Soprano_1 --song-parts 1 --max-hours 0.75

**There are two stages.** The first (heavy, GPU) runs ContentVec and RMVPE into `_cache/`; the
second (light, CPU) aligns, normalises and subsets, then writes the shard. `--from-cache` re-runs
only the second stage, so **ablating the alignment or the 256-dim subset never re-runs the heavy
models**. Re-running is **bit-identical**.

**Do not forget `--song-parts`.** The default takes the parent directory as the song name, so
deep layouts collapse every song into one name — and song names drive the song-level train/eval
split, so collapsing them leaks.

### Training

    uv run python -m train --config configs/svc_base.yaml \
      --data_dirs data/<speaker>... --run_name svc_base_01 --out_root log --device cuda

    # fine-tune to the target (with GAN)
    uv run python -m train --config configs/svc_target_ft_gan.yaml \
      --data_dirs data/ritsu --init_from ckpt_060000.pt --finetune \
      --run_name svc_ritsu_ft_gan_01 --out_root log --device cuda

**Reusing a `--run_name` silently resumes from that run's latest checkpoint.** Always rename for
a new experiment so you never overwrite a base checkpoint. Online `pitch_aug` is unavailable for
SVC because the features are precomputed (`train.py` stops explicitly).

### Conversion (inference)

    uv run python tools/svc_convert.py --wav <source.wav> --out out/<name> \
      --ckpt log/<run>/ckpt_015000.pt --manifest data/ritsu/manifest.json \
      --spk-id 0 --num-steps 16 --self-check --match-loudness --device cpu

- Pass **the manifest that checkpoint was trained with**. Normalisation statistics differ per run
  (measured `loudness_mean` of −4.6289 / −4.6554 / −4.7417), so **unifying them makes it a
  different experiment**.
- `--self-check` writes the **ceiling** (`*_vocoder_only.wav`). Metrics are read as a gap from it,
  so **always pass it**.
- **Pass `--match-loudness` for material you bring in** (see the table below).
- **Generate the ceiling once and reuse it** with `--ceiling-from <an existing output dir>`. The
  NHVSing ONNX contains a `RandomNormalLike` with no seed, so **its output changes run to run**
  for identical input — not even CPU plus deterministic mode makes it match. Records whose
  `convert.json` says `ceiling_comparable: false` cannot be compared to each other by ceiling gap.

### Evaluation

    uv run python tools/asr_cer.py           --dir out/<name> --language ja --device cpu
    uv run python tools/cer_breakdown.py     --cer out/<name>/cer.json --testset out/m5/testset.json
    uv run python tools/speaker_similarity.py --converted out/<name> \
      --target download/ritsu --unrelated .m0data/unrelated_ref --seconds 20 --device cpu
    uv run python tools/timing_metrics.py    --dir out/<name>
    uv run python tools/signal_quality.py    --dir out/<name> --device cpu
    uv run python tools/rtf.py --wav <vocal.wav> --ckpt <ckpt> --manifest <manifest> --device cpu

**Speaker similarity depends on both the encoder and the clip length.** Only ECAPA-TDNN at
**12 s or longer** passes our pre-registered calibration for singing, and
`tools/speaker_similarity.py` refuses shorter clips. **Re-run `tools/speaker_calibrate.py`
whenever you change the encoder.**

**Split CER by language before reading it.** `tools/cer_breakdown.py` reports no gap for a
language whose ceiling median exceeds 10% (material where orthographic variation would be
charged as content degradation) or which has fewer than 3 clips.

### Tests

    uv run python tools/smoke/run_smoke.py     # wires up every path (synthetic audio, ~3 min on GPU)
    uv run python -m unittest test_svc_model test_svc_preprocess test_svc_dataset test_svc_metrics
    uv run ruff check .

There are **514 unit tests**, none of which need heavy models or network access. `run_smoke.py`
feeds synthetic waveforms, so it **proves the wiring, never the quality**.

## Defaults and caveats (all measured)

| Item | Value / what to do | Evidence |
|---|---|---|
| Flow steps | **16** (`tools/svc_defaults.py`) | Chosen by sweep. Brightness ties with 1 step, but detail (ratio 0.755 → 0.957) and speaker identity (recovery 78.2% → 90.6%) are clearly better. The cost is negligible (flow RTF 0.044 → 0.065) |
| Transpose for low voices | **+7 semitones** (`SVC_TRANSPOSE_LOW_VOICE`) | Chosen by sweep. **The previous +12 is worse on both axes** (recovery 90.8% → 81.6%; CER gap from ceiling +12.5 → +27.0 pts). **Validated only for the Japanese low-voiced speaker.** Re-measured on 11 VocalSet male clips on 2026-09-18: **+12 is better for male sources** (recovery is indistinguishable; brightness lands 13 Hz from the target's own 1177 Hz, while +7 sits 332 Hz away). **The right amount depends on the source's F0, so no fixed constant covers both** |
| Material you bring in | **Pass `--match-loudness`** | Training takes features at the raw level. Streaming-ready material is pushed to near peak 1.0 (the Namine Ritsu DB peaks at 0.107), which drops brightness to −47% of the ceiling. **Even professional studio recordings need it** (0.71x → 0.93x) |
| Input level | **Never peak-normalise at inference** | Same reason. Normalising makes the model lift the low end and shave the highs (spectral centroid 620 → 368 Hz). **Content metrics do not catch this** (content cos moves only 0.8217 → 0.8096) |
| Vocoder version | **SVC stays on V3.1** (`checkpoints/nhv_v3_1.onnx`); V3.2 is bundled too | The ceiling moves with the vocoder, so adopting V3.2 means **re-measuring every "gap from ceiling" in this README**. V3.2 fixes **an abrupt per-frame weakening of the waveform at high pitch**, which **may be the defect listeners reported as a wavering level** — so the switch has to come with the re-measurement |
| Ceiling | **Build once, reuse via `--ceiling-from`** | The vocoder's output changes run to run. Measured: 6 of 26 clips shifted their ceiling CER by more than 5 points, because ASR is discrete and a tiny acoustic difference rewrites a transcript |

**Intelligibility is limited by data volume and diversity.** `num_steps`, the GAN, fine-tune
steps, base training length and **speaker familiarity** all leave it unchanged; **the only thing
that moves it is whether the song was seen in training** (+1.0 pts for training songs vs
+7.3 to +11.1 pts for unseen ones). **`eval/loss` is not a proxy** — continued pretraining took
it from 0.02311 to 0.01459 with essentially no change in intelligibility.

**The GAN does fix over-smoothing.** Fine-tuning with `configs/svc_target_ft_gan.yaml` raises
unseen-source speaker recovery from 69.4% to 75.3% with **no trade-off** — but **it does not move
intelligibility** (+13.2 vs +12.7 pts against the non-GAN fine-tune).

The Japanese research suite covering requirements, architecture, data/GPU, training, evaluation,
prior art/licensing, implementation status and sources is indexed at [doc/svc.md](doc/svc.md).

## Future work

- **Validation on male and low-voiced unseen speakers** — the two we measured are **female and
  in nearly the target's range**, and **time below C3** is still missing from every corpus.
- **Data volume and diversity** — that is where intelligibility is limited (above).
  **It remains possible that +12.7 pts is simply the level for this scale of data.**
- **The real-time student is not started** (removed from the serial path on 2026-09-14 and made
  conditional). **The vocoder is 93% of RTF**, so a faster acoustic model does not move
  end-to-end.
- **Multilingual support** — so far we validate with Japanese data, but the design itself is
  language-independent. We plan to support other languages with their own phoneme dictionaries
  and data, aiming for a single model that can handle multiple languages.
- **Higher quality** — we think there is still room to improve speaker fidelity, for example by
  refining how the pseudo-mel is generated and tuning its parameters.

## SVS path (phonemes + durations -> mel)

Everything below comes from the upstream project,
[LeapSinger](https://github.com/wavtechyukky/LeapSinger). **This path is untouched here**, and
the demo is upstream's.

LeapSinger is a diffusion-style **acoustic model** for singing voice. It runs very fast — even on a CPU — and produces stable, clean periodic (harmonic) content. It takes phonemes, their durations, and pitch (F0), and generates a mel-spectrogram. Training needs only audio plus phoneme labels with timing; it does not depend on any particular language. (All of our examples use Japanese data, though.)

![LeapSinger overview](doc/fig/leapsinger_overview.png)

**▶ Listen to the demo: https://wavtechyukky.github.io/LeapSinger/demo/**

### What is LeapSinger?

Most diffusion models start from random noise and paint the mel little by little over many steps. LeapSinger instead starts from a **"pseudo-mel" built from F0** (an impulse waveform plus white noise), and finishes the mel in **a single step** with a rectified flow. It skips the slow work of learning how to draw clean periodic content, and jumps straight to a high-quality mel — hence the name *LeapSinger*.

The pseudo-mel differs between the v/uv model and the non-v/uv model. The v/uv model can deliberately produce unvoiced regions.

![Pseudo-mel with and without v/uv](doc/fig/pseudo_mel_vuv.png)

The non-v/uv model (top) lays harmonics across every frame. The v/uv model (bottom) gates the harmonics off in unvoiced frames (the dark vertical bands are the unvoiced regions).

Through a lot of experiments we found two things: a high-quality neural vocoder for singing is sensitive to the *texture* of the mel, and the hard part for an acoustic model is drawing clean periodic content. LeapSinger starts not from noise but from a pseudo-mel that already has the shape of the pitch, so it avoids the hardest part — learning to draw the periodic content. This design gives:

- **High-quality periodic content** — it draws clean, low-noise periodic content stably. This helps both the texture of the vocoder output and how well each speaker's voice is reproduced.
- **Speed** — there is only one reverse step, so even on a single CPU core the RTF is under 0.03. You *can* use more steps, but going beyond one step actually moves the result *away* from the ground truth. (**That is the existing singing-synthesis (SVS) path.** The **SVC path defaults to 16 steps**, chosen by a sweep: brightness is the same as at one step, but fine detail and speaker identity are clearly better at 16. See `tools/svc_defaults.py`.)

Other features:

- **Multi-speaker** — switch voices by speaker ID. As an example, we distribute a model with three Japanese singers (Oniku Kurumi / Natsume Yuuri / Namine Ritsu).
- **Style control** — switch singing styles within the same speaker.

### Demo

**https://wavtechyukky.github.io/LeapSinger/demo/**

- A three-speaker model, comparing the generated results side by side with GT (the real recordings). Training on a single speaker slightly improves the speaker fidelity of the mel, but we saw no large difference in synthesis quality.
- A style-control demo.

### Performance

RTF (Real-Time Factor) measured on CPU. Smaller is faster; below 1 means faster than real time.

RTF per acoustic model, comparing Python native vs. ONNX across core counts:

| Cores | Python native | ONNX |
|:--:|--:|--:|
| 1 | 0.027 | 0.090 |
| 2 | 0.026 | 0.063 |
| 4 | 0.027 | 0.058 |
| 8 | 0.026 | 0.054 |
| 10 | 0.024 | 0.065 |

- **Python native** is a single step, so it barely depends on core count. Even on one core the RTF is 0.027 (about 37× faster than real time).
- **ONNX** is a few times slower than native because of onnxruntime overhead, but it is still more than 10× faster than real time. 4–8 cores are fastest; using all cores (10) is actually slower.
- **The NHVSing vocoder** runs at RTF under 0.1 on CPU (see the NHVSing repository for details).

(Measured on Apple Silicon, 10 cores, onnxruntime CPU, a ~7-second phrase, median. Results vary by machine.)

**This table is the SVS acoustic model alone.** The **SVC path's end-to-end cost is a different quantity** (see the RTF section above): measured at **total RTF 0.464 on GPU, of which the vocoder is 0.432 (93%)** and the acoustic model only 0.006 (0.654 total on CPU). **Speeding up the acoustic model barely moves end-to-end.** We also **do not call it "real-time" even when `realtime_capable` is True** — chunk boundaries, audio I/O and sustained operation are unmeasured.

The frame settings are 44.1 kHz and hop size 256. Hop size 512 is handled by averaging each pair of adjacent frames.

### Architecture

The flow in the figure above is:

1. **Input** — phonemes, durations, F0 (plus speaker ID if needed).
2. **Encoder** — embed the phonemes, stretch them to frame length according to the durations, and add F0 and speaker to form the condition.
3. **Harmonic + Noise Excitation** — build a "pseudo-mel" from F0 (an impulse waveform plus white noise). This is the starting point of the flow.
4. **Rectified Flow (1 step)** — starting from the pseudo-mel and conditioned on the condition, transform it into a realistic mel in a single step.
5. **NHVSing** — a high-quality neural vocoder that turns the mel into audio (a waveform), at RTF under 0.1 on CPU. https://github.com/wavtechyukky/NHVSing/

The pseudo-mel lets you tune the harmonic decay, the number of harmonics, and the strength of the white noise. That said, after much testing, stacking the impulse's harmonics all the way up to the Nyquist frequency gives the best quality.

### Usage

#### Setup

Python 3.13 only (pinned both by `requires-python` in `pyproject.toml` and by `.python-version`). Dependencies and execution go through [uv](https://docs.astral.sh/uv/).

    git clone https://github.com/ayutaz/LeapSVC
    cd LeapSVC
    uv sync                               # inference, re-synthesis, notebooks
    uv sync --extra train                 # training (adds TensorBoard)
    uv sync --extra export                # ONNX export
    uv sync --extra train --extra export  # everything

Run Python with `uv run python ...` and add dependencies with `uv add <package>` (do not use bare `python` / `pip`, or `uv pip`). `uv sync` fetches the interpreter named in `.python-version` and builds `.venv/` exactly as resolved in `uv.lock`.

- **PyTorch** is pinned to the CUDA wheel index (cu130) via `[[tool.uv.index]]` in `pyproject.toml`, so `uv sync` alone installs the GPU build on Windows / Linux (the Windows `torch` on PyPI is a CPU build, hence the explicit index). To use a different CUDA, change `cu130` in that url to `cu126` / `cu128` / `cu132` and re-run `uv lock`. On macOS a marker falls back to the CPU/MPS build from PyPI.
- To check the GPU is visible:

      uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
- **F0 extraction (RMVPE)** weights download automatically on first run (HuggingFace → `preprocess/algorithms/rmvpe.pt`).
- **The vocoder (NHVSing)** is bundled as ONNX under `checkpoints/`; no extra download is needed.
- **The acoustic model itself is distributed via Releases** (not included in the repo).

#### Configuration

Training and export are driven by YAML (there are examples in `configs/`). A YAML file has six sections: `mel` / `model` / `excitation` / `train` / `gan` / `data`. The main keys are:

- `model` — `spk_dim` (greater than 0 means multi-speaker), `n_speakers`, `n_styles` (greater than 0 enables styles), `use_uv` (whether to feed v/uv as a condition).
- `excitation` — `n_harm` (number of harmonics), `harm_decay` (harmonic decay), `noise_ratio` (white-noise strength).
- `train` — `lr`, `max_updates`, `num_steps` (inference steps; **1** recommended), `balance_speakers` (sample speakers equally), and so on.
- `gan` — settings for the GAN that sharpens texture. `enabled` (`false` = flow loss + mel loss only), `gan_start_step` (the step at which the GAN turns on), `gan_strength` (strength of the adversarial loss), and so on.
- `data` — `spk_map` / `style_map` (mapping from dataset folder name → speaker ID / style ID).

#### Building a dictionary

The Japanese phoneme list is in `dict/ja.phonemes` (one phoneme per line, the order is the ID, `pau` at the top is ID 0, and anything after `#` is a comment). To use a different language or your own phonemes, make a file in the same format and pass it to each command with `--phonemes <file>` (if omitted, the Japanese `dict/ja.phonemes` is used).

#### Preprocessing

Prepare one YAML per dataset (`configs/recipes/<db>.yaml`). Each song uses an audio `wav` and a `.lab` with phoneme timing (plus a score). The following command produces preprocessed data under `data/<db>/`.

    uv run python -m preprocess.run --recipe configs/recipes/<db>.yaml

The three databases used in the examples can be downloaded with these scripts (please follow each database's terms of use).

    uv run python preprocess/download_scripts/download_oniku.py
    uv run python preprocess/download_scripts/download_natsume.py
    uv run python preprocess/download_scripts/download_ritsu.py

F0 is extracted with RMVPE. (Please do not run RMVPE across multiple processes.)

#### Training

Training has two stages. The first stage learns the base voice with a flow loss plus a mel loss; the second stage turns on a GAN to sharpen texture. (This is set in the config's `gan` section; `gan.enabled: false` gives the flow loss plus mel loss only.)

    uv run python -m train --config configs/<name>.yaml \
      --data_dirs data/<db> [data/<db2> ...] \
      --run_name <name> --out_root log --device cuda

Running the same command again automatically resumes from where it stopped.

#### Export

    uv run python -m export.cli \
      --ckpt log/<run>/ckpt_050000.pt \
      --out export/<name> --model-name <name> \
      --variant diffsinger --hop 256 --speaker bake --spk-id 0

For speaker handling (bake / embed / none) and other details, see "Export to ONNX" below.

You can try the whole flow — from export to use — in a notebook. Download the model from the Release and place it in `notebooks/sample_data/` (see `place_model_here.txt` in that folder).

    notebooks/export_and_use_onnx.ipynb

The notebook exports the acoustic model to a single ONNX and runs it end to end (phonemes + duration + F0 → mel → audio).

#### Export to ONNX

`export/` converts a checkpoint into a self-contained ONNX graph. The excitation and the single-step flow are baked into the graph, so the caller only needs to pass phonemes, durations, and F0. For speaker handling, you can choose:

- **bake** — fix a single voice. This makes the simplest graph (no speaker input).
- **embed** — take a speaker vector as input. One graph can switch to any voice.
- **none** — for single-speaker models. No speaker input and no baking (a graph with no notion of speaker). For multi-speaker models, use bake or embed.

#### Vocoder

Two NHVSing vocoders are bundled under `checkpoints/`.

- `nhv_v3_2.onnx` — takes a hop-size-256 mel and F0.
- `nhv_v3_2x.onnx` — takes a hop-size-512 mel and F0.

`nhv_v3_1.onnx` and `nhv_v3_1x.onnx` (V3.1) are kept alongside them, because **every SVC
measurement is relative to a V3.1 ceiling** and cannot be restated until it is re-measured.

The default is **V3.2** — the latest weights. It fixes an abrupt per-frame weakening of the waveform at high pitch by switching the LTV filter's overlap-add to a Hann window. The input/output contract is identical to V3.1, so it is a drop-in replacement (see [NHVSing](https://github.com/wavtechyukky/NHVSing/) for details).

### Options

- **Dictionary** — you can specify any phoneme dictionary. The design can handle languages other than Japanese (multilingual support itself is future work).
- **v/uv handling** — choose between a mode that feeds voiced/unvoiced (v/uv) as a condition, and a mode that uses only a continuous F0 with the gaps filled by linear interpolation.
- **Excitation tuning** — change the harmonic decay, the number of harmonics, and the white-noise strength.
- **Training recipes** — training conditions are set in YAML (which datasets to use, speaker IDs, per-speaker styles and data, and so on).

## License

The code is MIT (`LICENSE`). However, the bundled vocoder ONNX files (`checkpoints/nhv_v3_2*.onnx`), the trained models distributed via Releases, and the singing databases used to train them are **not** covered by MIT — they follow their own licenses and terms of use (see the Acknowledgments below and `CREDITS.txt` in the model release).

**The SVC path is more restricted.** Its base model is trained on **GTSinger (CC BY-NC-SA 4.0 — non-commercial, ShareAlike)**, and whether ShareAlike reaches trained weights is not settled by the license text. **No SVC weights are distributed**: the project decision is research and personal use only (`doc/svc-dataset-ledger.md`). GTSinger's README also forbids generating a specific person's singing voice without their consent, so **converting a voice requires the target singer's consent**, independently of any software license. See the SVC NOTICE in `LICENSE`.

## Acknowledgments

Thanks to the datasets used to train this model, and to the related projects.

- Oniku Kurumi singing database (Oniku Kurumi) — https://onikuru.info/db-download/
- Natsume Yuuri (database production: アマノケイ / voice provider: 霧野蒼太) — https://ksdcm1ng.wixsite.com/njksofficial/enunu-nnsvs
- Namine Ritsu — https://www.canon-voice.com/voicebanks/
- Neural Homomorphic Vocoder — https://www.isca-archive.org/interspeech_2020/liu20_interspeech.html
- dsp (zjlww) — https://github.com/zjlww/dsp

The SVC path additionally uses:

- GTSinger (CC BY-NC-SA 4.0) — https://github.com/AaronZ345/GTSinger
- VocalSet (CC BY 4.0; unseen-source evaluation only) — https://zenodo.org/records/1492453
- ContentVec (MIT) — https://huggingface.co/lengyue233/content-vec-best
- RMVPE — https://arxiv.org/abs/2306.15412 (weights fetched from lj1995/VoiceConversionWebUI; **license not verified**)
- Tohoku Kiritan (©SSS; **unseen-speaker evaluation only** — not used for training) — https://zunko.jp/kiridev/login.php
- No.7 (derived from the Kotori Koiwai singing database; songs belong to Kotori Koiwai, commercial use via the No.7 production committee; **unseen-speaker evaluation only** — not used for training) — https://voiceseven.com/

The distributed multi-speaker models display the credits above, following each database's terms. For Natsume Yuuri, we display **database production: アマノケイ / voice provider: 霧野蒼太**, and we bundle the "Terms of use for Natsume Yuuri's output audio" with the model distribution.
