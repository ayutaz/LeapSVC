#!/usr/bin/env python3
"""音量の揺れを測る（[実行計画](../doc/svc-plan.md) M5 ゴール 4 の続き）。

    uv run python tools/loudness_stability.py --dir out/m5/leapsvc_gan_v31 \
      --out out/m5/metrics_gan_v31_fixed/loudness.json

**blind で名前が付いた defect を客観化するための指標です。** 2026-09-13 の聴取で、
LeapSVC の出力について 6 本中 5 本が「**音量が揺れる**」を挙げました。preference は
「どちらが良いか」しか残さないので、名前が付いてから測るという順序になります。

**source を基準にします。** 変換後の音量の**動き**が入力の歌い方をなぞっているかを見るので、
上限（`*_vocoder_only.wav`）が要りません。つまり **baseline にもそのまま当たり、
系をまたいで比べられます**（[CLAUDE.md](../CLAUDE.md) の「上限を共有できない指標は
系をまたげない」の裏返し）。

| 量 | 意味 |
|---|---|
| `residual_db_std` | source との log 音量の差から**平均を引いた**残差の標準偏差（dB）。**これが揺れ** |
| `envelope_corr` | 音量の動きの相関。1 に近いほど入力の抑揚をなぞっている |
| `slow_db_std` / `fast_db_std` | 残差を 2 Hz で分けたもの。聴取で名前が付いたのは速い側 |
| `frame_jitter_db` | **隣り合うフレームの差**の標準偏差。`fast_db_std` は 2 Hz より速い成分すべてなので、数 Hz の抑揚のずれと **1 フレームの欠損**を区別しません（NHVSing V3.2 が直した defect の形）。**選択性は 1.6 倍しかありません**（合成信号の実測。疎な欠損は std に薄まる）ので、**この量だけで defect の有無を決めないこと** ―― 従来の指標と**順序が反転する**ことだけが根拠になります |

**全体の音量差は揺れではありません。** 変換の結果として全体が大きく／小さくなるのは
別の話なので、平均を引いてから見ます。**無音は使いません** ―― log-RMS が極端に小さくなり、
残差を支配してしまうためです。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# 無音の除外。source の最大フレームからこれだけ下がったフレームは使わない。
SILENCE_FLOOR_DB = 40.0
# 絶対の下限。相対だけでは **全部無音の clip を見分けられません**（最大も無音なので
# 「最大から 40 dB 下」が全フレームを通してしまい、残差 0 を「揺れていない」と誤読します）。
ABS_FLOOR_DB = -70.0
_EPS = 1e-10


def loudness_envelope(x, *, sr: int, hop: int):
    """フレームごとの log 音量（dB）。窓は hop の 2 倍、ホップは `hop`。"""
    import numpy as np

    x = np.asarray(x, dtype=np.float32).ravel()
    win = hop * 2
    if len(x) < win:
        return np.zeros(0, dtype=np.float32)
    n = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    frames = x[idx]
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + _EPS)
    return (20.0 * np.log10(rms + _EPS)).astype(np.float32)


# ここより遅い変動は「フレーズの作り方」、速い変動を「揺れ」とみなす境目（Hz）。
SLOW_CUTOFF_HZ = 2.0


def _smooth_frames(sr: int, hop: int) -> int:
    """`SLOW_CUTOFF_HZ` に相当する移動平均の窓（フレーム数、奇数）。"""
    fps = sr / hop
    w = max(3, int(round(fps / SLOW_CUTOFF_HZ)))
    return w + 1 - w % 2


def _moving_average(x, w: int):
    """端を複製して長さを保つ移動平均。**端で窓が縮むと、そこだけ揺れて見えます。**"""
    import numpy as np

    if w <= 1 or len(x) < 3:
        return np.asarray(x, dtype=np.float64)
    w = min(w, len(x) - (1 - len(x) % 2))
    if w < 3:
        return np.asarray(x, dtype=np.float64)
    pad = w // 2
    xp = np.pad(np.asarray(x, dtype=np.float64), pad, mode="edge")
    return np.convolve(xp, np.ones(w) / w, mode="valid")


def loudness_report(source, converted, *, sr: int, hop: int,
                    floor_db: float = SILENCE_FLOOR_DB) -> dict[str, Any]:
    """source に対する変換後の音量の揺れ。

    **平均を引いてから残差を見ます。** 全体の音量差（gain）は揺れではありません。
    **鳴っているフレームだけを使います** ―― 無音の log-RMS は極端に小さく、残差を
    支配してしまいます。使えるフレームが無ければ `None` を返します（0 ではありません。
    「揺れていない」と区別が付かなくなるため）。
    """
    import numpy as np

    a = loudness_envelope(source, sr=sr, hop=hop)
    b = loudness_envelope(converted, sr=sr, hop=hop)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    out: dict[str, Any] = {"n_frames_total": int(n), "n_frames_used": 0,
                           "residual_db_std": None, "envelope_corr": None,
                           "gain_db": None, "slow_db_std": None, "fast_db_std": None,
                           "frame_jitter_db": None}
    if n == 0:
        return out

    keep = (a > (a.max() - floor_db)) & (a > ABS_FLOOR_DB)
    out["n_frames_used"] = int(keep.sum())
    if keep.sum() < 8:
        return out

    d = (b[keep] - a[keep]).astype(np.float64)
    out["gain_db"] = float(d.mean())
    out["residual_db_std"] = float(d.std())
    # **「揺れる」は速さの話。** ゆっくりした抑揚のずれ（フレーズの作り方の違い）と、
    # 伸ばした音の中で細かく上下する揺れを分ける。聴取で名前が付いたのは後者。
    slow = _moving_average(d, _smooth_frames(sr, hop))
    out["slow_db_std"] = float(slow.std())
    out["fast_db_std"] = float((d - slow).std())
    # **フレーム単位の揺れ。** `fast_db_std` は 2 Hz より速い成分すべてなので、
    # 数 Hz の抑揚のずれと 1 フレームの欠損を区別しません。隣り合うフレームの差を
    # 取ると後者にだけ強く反応します（NHVSing V3.2 が直した defect の形）。
    # **無音を挟んだ差は取りません** ―― そこだけ巨大な差になり、全体を支配します。
    full = (b - a).astype(np.float64)
    pair = keep[:-1] & keep[1:]
    if pair.sum() >= 8:
        out["frame_jitter_db"] = float(np.diff(full)[pair].std())
    av, bv = a[keep].astype(np.float64), b[keep].astype(np.float64)
    if av.std() > 1e-9 and bv.std() > 1e-9:
        out["envelope_corr"] = float(np.corrcoef(av, bv)[0, 1])
    return out


def main() -> int:
    import argparse
    import json
    import statistics as st

    import numpy as np
    import soundfile as sf

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True,
                    help="`*_source.wav` と `*_converted.wav` が並ぶディレクトリ")
    ap.add_argument("--hop", type=int, default=256)
    ap.add_argument("--floor-db", type=float, default=SILENCE_FLOOR_DB)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    def read(p: Path):
        w, sr = sf.read(str(p), dtype="float32", always_2d=False)
        if w.ndim > 1:
            w = w.mean(axis=1)
        return np.ascontiguousarray(w, dtype=np.float32), sr

    rows = []
    for src in sorted(Path(a.dir).glob("*_source.wav")):
        stem = src.name[: -len("_source.wav")]
        cnv = src.with_name(f"{stem}_converted.wav")
        if not cnv.exists():
            continue
        sw, sr = read(src)
        cw, cr = read(cnv)
        if sr != cr:
            sys.exit(f"{stem}: sample rate が違います（{sr} / {cr}）")
        r = loudness_report(sw, cw, sr=sr, hop=a.hop, floor_db=a.floor_db)
        r["file"] = stem
        r["tag"] = stem.split("__")[-1]
        rows.append(r)
        std = "n/a" if r["residual_db_std"] is None else f"{r['residual_db_std']:5.2f}"
        cor = "n/a" if r["envelope_corr"] is None else f"{r['envelope_corr']:.3f}"
        print(f"  {r['tag']:10s} 残差 {std} dB  相関 {cor}")

    if not rows:
        sys.exit(f"{a.dir}: `*_source.wav` と `*_converted.wav` の組がありません")

    usable = [r for r in rows if r["residual_db_std"] is not None]
    summary = {
        "n_clips": len(rows), "n_usable": len(usable),
        "residual_db_std": st.median(r["residual_db_std"] for r in usable) if usable else None,
        "envelope_corr": st.median(r["envelope_corr"] for r in usable
                                   if r["envelope_corr"] is not None) if usable else None,
    }
    print("\n=== 音量の揺れ（中央値）===")
    print(f"  残差 {summary['residual_db_std']:.2f} dB <- 大きいほど揺れている")
    print(f"  抑揚の相関 {summary['envelope_corr']:.3f}")
    print("\n  ** source 基準なので baseline にもそのまま当たります（上限は要りません） **")

    rep = {"summary": summary, "clips": rows}
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
