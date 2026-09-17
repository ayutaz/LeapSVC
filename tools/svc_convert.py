#!/usr/bin/env python3
"""任意の WAV を学習済み SVC モデルで変換する。

    uv run python tools/svc_convert.py --wav path/to/vocal.wav --out out/ \\
      --ckpt .m0data/m3/ckpt_030000.pt --manifest .m0data/m3/m3_out/manifests/ritsu/manifest.json \\
      --spk-id 22 --device cpu

**入力はボーカルのみ（伴奏なし）の音声**であること。伴奏が混ざっていると ContentVec も RMVPE も
歌声以外を拾い、変換結果が崩れます。長い曲は `--chunk-sec` ごとに分割して変換し、つなげます。

正規化は **target 話者の manifest の統計**を当てます。shard は話者ごとに自分の統計で
正規化されているので、その話者の音量スケールでモデルが学習しているためです。
`features_to_item()` が学習時と同じ手順を再現するので、ここでずれることはありません。

`--self-check` を付けると、入力自身の ground-truth mel も NHVSing に通して並べて書き出します。
**変換の劣化のうち、どこまでがボコーダー由来かを切り分ける**ための対照です。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

if os.environ.get("LEAPSINGER_NONDETERMINISTIC") != "1":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from leapsinger.config import MelSpec  # noqa: E402
from tools.audio_metrics import band_profile, format_profiles  # noqa: E402
from tools.svc_defaults import SVC_NUM_STEPS  # noqa: E402

_TAG_SAFE = re.compile(r"[^\w-]")


def output_stem(wav_path: str, *, tag: str | None) -> str:
    """出力ファイルの stem。**`tag` で区間を区別できるようにする。**

    同じ WAV から `--start` を変えて 2 回変換すると、入力名だけでは衝突します。hold-out は
    1 曲から複数区間を取るのでこれが実際に起きます。**黙って上書きされると clip 数が減る
    だけで例外にならない**ので、呼び出し側が区別できる名前を渡せるようにします。
    """
    stem = Path(wav_path).stem
    if not tag:
        return stem
    return f"{stem}__{_TAG_SAFE.sub('_', str(tag))}"


# `torch.use_deterministic_algorithms(True)` を CUDA で使うときに要る値（torch の要求）。
_CUBLAS_OK = (":4096:8", ":16:8")


def deterministic_env_error(device: str, env) -> str | None:
    """決定的モードを CUDA で使うのに環境変数が足りないなら、その説明を返す。

    **変換を走らせてから torch に落とされないため、起動時に確かめます。**
    CPU は既定で決定的なので何も要りません。
    """
    if not str(device).startswith("cuda"):
        return None
    v = dict(env).get("CUBLAS_WORKSPACE_CONFIG")
    if v in _CUBLAS_OK:
        return None
    return ("--deterministic を CUDA で使うには CUBLAS_WORKSPACE_CONFIG が要ります"
            f"（{' か '.join(_CUBLAS_OK)}。いまは {v!r}）。"
            "これが無いと torch が実行時に落ちます")


def provenance(*, ckpt: str, vocoder: str, device: str, deterministic: bool,
               env, ceiling_from: str | None = None) -> dict:
    """上限が比べられるかを判断するための記録。

    **NHVSing の ONNX は seed 属性の無い `RandomNormalLike` を持っています**
    （`node_randn_like`。属性は `scale` と `mean` だけ）。ONNX Runtime は毎回別の乱数を
    引くので、**ボコーダーを通した出力は run ごとに必ず変わります** ―― 上限も変換結果も
    です。**device も `use_deterministic_algorithms` も関係ありません**（CPU + 決定的モードの
    2 回の run で sha256 が一致しないことを実測しました）。

    これが、同じ V3 ボコーダーの 2 つの run で **26 clip 中 6 本の上限 CER が 5 点を超えて
    ずれた**（最大 88 点）本当の原因です。当初は CUDA の非決定性を疑いましたが誤りでした。

    **唯一の対処は、上限を 1 度だけ作って系をまたいで使い回すことです**（`ceiling_from`）。
    `ceiling_comparable` が False の記録どうしは、**上限が違うので「上限との差」を
    比べられません**。
    """
    cublas = dict(env).get("CUBLAS_WORKSPACE_CONFIG")
    return {
        "ckpt": ckpt,
        # **ボコーダーを記録する。** 変わると上限が変わるのに、記録が無かった。
        "vocoder": vocoder,
        "device": device,
        # acoustic 側（flow と特徴抽出）だけを決定的にする。**出力 WAV は
        # ボコーダーの乱数があるので、これでも bit 一致しません。**
        "deterministic": bool(deterministic),
        "cublas_workspace_config": cublas,
        # NHVSing の性質。**seed を渡す口が無い**ので、事実として記録しておく。
        "vocoder_stochastic": True,
        "ceiling_source": str(ceiling_from) if ceiling_from else "generated",
        "ceiling_comparable": bool(ceiling_from),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wav", required=True, help="変換する WAV（ボーカルのみ）")
    ap.add_argument("--out", required=True, help="出力ディレクトリ")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True, help="target 話者の shard の manifest.json")
    ap.add_argument("--spk-id", type=int, default=0, help="変換先の speaker id")
    ap.add_argument("--num-steps", type=int, default=SVC_NUM_STEPS,
                    help=f"flow の step 数（既定 {SVC_NUM_STEPS}。2026-09-01 の掃引で決定）")
    ap.add_argument("--transpose", type=float, default=0.0,
                    help="F0 を半音単位で移調する。男女をまたぐときに使う")
    ap.add_argument("--chunk-sec", type=float, default=20.0,
                    help="この長さごとに分けて変換する（長い曲のメモリ対策）")
    ap.add_argument("--tag", default=None,
                    help="出力名に付ける識別子。**同じ WAV から区間を変えて変換するときは必須**"
                         "（付けないと前の結果を黙って上書きします）")
    ap.add_argument("--start", type=float, default=0.0, help="この秒数から")
    ap.add_argument("--seconds", type=float, default=0.0, help="この長さだけ（0 なら全部）")
    ap.add_argument("--peak-normalize", action="store_true",
                    help="入力を peak 0.95 へ揃える。**学習と条件が変わるので既定は off**")
    ap.add_argument("--match-loudness", action="store_true",
                    help="入力の音量を **学習分布の平均へ寄せる**。配信用に整えられた音源は"
                         "学習素材よりずっと大きく、そのままだと高域が削れる（実測 -47%%、"
                         "合わせると -22%%）。手元録音や配信音源にはこれを付ける")
    ap.add_argument("--self-check", action="store_true",
                    help="入力の GT mel も NHVSing に通し、ボコーダー由来の劣化を分離する")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--content-model", default="lengyue233/content-vec-best")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--vocoder", default="checkpoints/nhv_v3_1.onnx")
    ap.add_argument("--ceiling-from", default=None,
                    help="**上限（`*_vocoder_only.wav`）を作り直さず、このディレクトリから"
                         "使い回す。** NHVSing は seed の無い RandomNormalLike を持つので、"
                         "上限は作るたびに変わります（実測で上限 CER が 26 clip 中 6 本、"
                         "最大 88 点ずれた）。**系を比べる測定では必ず指定すること**")
    ap.add_argument("--deterministic", action="store_true",
                    help="acoustic 側（特徴抽出と flow）を決定的にする。**出力 WAV は"
                         "これでも bit 一致しません** ―― NHVSing が seed の無い乱数を"
                         "持つためで、上限を揃えるには --ceiling-from が要ります。"
                         "CUDA では CUBLAS_WORKSPACE_CONFIG=:4096:8 も必要")
    a = ap.parse_args()

    # **`CUBLAS_WORKSPACE_CONFIG` はこの module の import 時に既に立てています**（上部）。
    # ただし**それだけでは決定的になりません** ―― `use_deterministic_algorithms(True)` が
    # 要ります。片方だけ入っていたのが、上限が run ごとに変わっていた原因です。
    err = deterministic_env_error(a.device, os.environ) if a.deterministic else None
    if err:
        sys.exit(err)

    import soundfile as sf

    from infer import infer_svc_mel, load_acoustic, load_vocoder, mel_to_wav
    from preprocess.svc.encoders import ContentVecEncoder, RmvpeF0
    from preprocess.svc.extract import _resample, extract_phrase, transpose_f0
    from preprocess.svc.loudness import frame_log_rms, loudness_match_gain
    from preprocess.svc.shard import features_to_item

    mel = MelSpec()
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(a.manifest).read_text(encoding="utf-8"))
    if a.deterministic:
        import torch

        torch.use_deterministic_algorithms(True)
        print("[convert] --deterministic: acoustic 側を決定的にします"
              "（**ボコーダーの乱数は残るので出力 WAV は bit 一致しません**）", flush=True)
    model, cfg = load_acoustic(a.ckpt, device=a.device)
    if not 0 <= a.spk_id < max(1, int(cfg.get("n_speakers", 1))):
        sys.exit(f"spk_id {a.spk_id} は 0..{cfg.get('n_speakers', 1) - 1} の範囲外です")
    vocoder = load_vocoder(str(ROOT / a.vocoder), sr=mel.sr)
    encoder = ContentVecEncoder(a.content_model, layer=a.layer, device=a.device)
    f0x = RmvpeF0(device=a.device)
    print(f"[convert] arch={cfg.get('arch')} n_speakers={cfg.get('n_speakers')} "
          f"-> spk_id={a.spk_id}  steps={a.num_steps}  device={a.device}", flush=True)

    wav, sr = sf.read(a.wav, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = _resample(np.ascontiguousarray(wav, dtype=np.float32), sr, mel.sr)
    if a.start or a.seconds:
        a0 = int(a.start * mel.sr)
        a1 = a0 + int(a.seconds * mel.sr) if a.seconds else len(wav)
        wav = np.ascontiguousarray(wav[a0:a1])
    # **音量を触らない。** 学習時（`preprocess.svc.run`）は生の音量のまま特徴を取るので、
    # ここで peak 正規化すると loudness 条件が学習分布からずれる。実測では 1.2 sigma 上へ
    # ずれ、モデルが低域を持ち上げて高域を削り、mel L1 が 0.56 -> 0.73 に悪化した。
    # 試聴用の source だけは書き出す直前に揃える（特徴抽出には使わない）。
    peak = float(np.abs(wav).max())
    print(f"[convert] {Path(a.wav).name}  {len(wav) / mel.sr:.1f}s  {mel.sr} Hz  "
          f"peak {peak:.3f}（正規化しない）", flush=True)
    if a.peak_normalize:
        wav = (wav / max(peak, 1e-9) * 0.95).astype(np.float32)
        print("[convert] --peak-normalize: 学習と条件が変わる。比較には使わないこと", flush=True)
    if a.match_loudness:
        g = loudness_match_gain(wav, manifest, hop=mel.hop, n_fft=mel.n_fft)
        before = frame_log_rms(wav, hop=mel.hop, n_fft=mel.n_fft).mean()
        wav = (wav * g).astype(np.float32)
        after = frame_log_rms(wav, hop=mel.hop, n_fft=mel.n_fft).mean()
        sd = float(manifest.get("loudness_std", 1.0)) or 1.0
        print(f"[convert] --match-loudness: x{g:.3f}  loudness {before:+.3f} -> {after:+.3f} "
              f"(学習分布から {(before - manifest['loudness_mean']) / sd:+.2f} -> "
              f"{(after - manifest['loudness_mean']) / sd:+.2f} sigma)", flush=True)

    step = int(a.chunk_sec * mel.sr)
    pieces, gt_pieces, t0 = [], [], time.time()
    for i, s in enumerate(range(0, len(wav), step)):
        seg = np.ascontiguousarray(wav[s:s + step])
        if len(seg) < mel.hop * 4:
            break
        feats = extract_phrase(seg, mel.sr, content_encoder=encoder, f0_extract=f0x, mel=mel)
        feats["f0_hz"] = transpose_f0(feats["f0_hz"], a.transpose)
        item = features_to_item(feats, manifest)
        item["spk_id"] = int(a.spk_id)
        pred = infer_svc_mel(model, item, num_steps=a.num_steps, device=a.device)
        logf0 = np.log2(np.maximum(feats["f0_hz"], 1.0))
        pieces.append(mel_to_wav(vocoder, pred, logf0, feats["uv"]))
        # **上限を使い回すときは作らない。** ボコーダーは最も重い段（GPU でも合計 RTF の
        # 93%）なので、捨てる音を合成するのは丸損です。
        if a.self_check and not a.ceiling_from:
            gt_pieces.append(mel_to_wav(vocoder, feats["mel"], logf0, feats["uv"]))
        print(f"  chunk {i}: {len(seg) / mel.sr:.1f}s  voiced {np.mean(feats['uv'] > 0.5):.2f}",
              flush=True)

    if not pieces:
        sys.exit("変換できる長さがありませんでした")
    conv = np.concatenate(pieces).astype(np.float32)
    stem = output_stem(a.wav, tag=a.tag)
    sf.write(out_dir / f"{stem}_converted.wav", conv, mel.sr)
    # 試聴用の source は聴きやすさのために揃える（特徴抽出には使っていない）
    _p = float(np.abs(wav).max())
    sf.write(out_dir / f"{stem}_source.wav",
             (wav / _p * 0.95).astype(np.float32) if _p > 1e-6 else wav, mel.sr)
    report = {"wav": str(a.wav), "spk_id": a.spk_id, "num_steps": a.num_steps,
              **provenance(ckpt=a.ckpt, vocoder=a.vocoder, device=a.device,
                           deterministic=a.deterministic, env=os.environ,
                           ceiling_from=a.ceiling_from),
              "seconds": len(wav) / mel.sr, "elapsed_sec": round(time.time() - t0, 1),
              "source": band_profile(wav, mel.sr), "converted": band_profile(conv, mel.sr)}
    if a.self_check:
        dst = out_dir / f"{stem}_vocoder_only.wav"
        if a.ceiling_from:
            # **作り直さずに使い回す。** ボコーダーの乱数で毎回変わるため。
            src_ceil = Path(a.ceiling_from) / dst.name
            if not src_ceil.exists():
                sys.exit(f"--ceiling-from に {dst.name} がありません（{src_ceil}）。"
                         "**黙って作り直すと、系をまたいで別の上限を比べることになります**")
            shutil.copyfile(src_ceil, dst)
            gt, _ = sf.read(str(dst), dtype="float32", always_2d=False)
            print(f"[convert] 上限を使い回しました: {src_ceil}", flush=True)
        else:
            gt = np.concatenate(gt_pieces).astype(np.float32)
            sf.write(dst, gt, mel.sr)
            print("[convert] **注意: この上限は作り直したものです。** NHVSing の乱数で "
                  "run ごとに変わるので、**系を比べる測定には使えません**"
                  "（--ceiling-from で使い回すこと）", flush=True)
        report["vocoder_only"] = band_profile(np.asarray(gt, dtype=np.float32), mel.sr)
    (out_dir / f"{stem}_convert.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n" + format_profiles({k: report[k] for k in
                                  ("source", "vocoder_only", "converted") if k in report}))
    print(f"\n-> {out_dir}  ({report['elapsed_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
