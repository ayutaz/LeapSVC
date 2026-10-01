"""入力の**歌唱音量からの相対値**で、無音フレームを無声にする（[実行計画](../../doc/svc-plan.md) 13.5）。

`extract_f0_rmvpe` は chunk ごとにピークで正規化してから RMVPE に渡すので、無音の chunk では
録音に乗った持続音が 40〜60 dB 持ち上げられ、RMVPE が有声と判定します。前処理の「有声率 0.3
未満の chunk を捨てる」をすり抜け、**くるみの chunk の 10% が F0 付きの無音として学習に
入っていました**（13.1 の訂正）。

**絶対音量のしきい値は使えません**（13.3 で実測）。録音ごとに音量が 20 dB 以上違い、大きい
録音の無音と小さい録音の歌声が重なるためです。基準は**入力全体の歌唱音量**（50 ms 窓 RMS の
90 パーセンタイル）で、そこから `gap_db` 以上小さいフレームを無声にします。前処理では曲ファイル、
推論では変換する入力全体が基準です ―― **chunk だけを見ても、それが無音かどうかは分かりません。**
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

WIN_SEC = 0.05
Q = 90.0
# 歌唱音量からこれ以上小さいフレームを無声にする（dB）。doc/svc-plan.md 13.5: 事前登録した規則では
# 無音側 p1 55.8 dB と歌唱側（くるみ）p99.9 56.5 dB が 0.7 dB 重なったが、くるみの裾は同じ持続音の
# 第 3 倍音（約 142 Hz）と見られ、除けば歌唱側の最大は 46.5 dB。利用者の決定で規則を覆して中点の 51 dB。
SILENCE_GAP_DB = 51.0
_TO_DB = 20.0 / np.log(10.0)


def singing_level_db(wav: np.ndarray, sr: int) -> float:
    """入力全体の歌唱音量（50 ms 窓 RMS の 90 パーセンタイル、dBFS）。"""
    w = np.asarray(wav, dtype=np.float64).reshape(-1)
    n = max(1, int(WIN_SEC * int(sr)))
    fr = w[: len(w) // n * n].reshape(-1, n)
    if not len(fr):
        fr = w.reshape(1, -1)
    return float(np.percentile(20.0 * np.log10(np.sqrt((fr ** 2).mean(axis=1)) + 1e-12), Q))


def frame_db(wav: np.ndarray, *, hop: int, n: int) -> np.ndarray:
    """フレームの RMS（dBFS）を `[n]` で返す。loudness 特徴と同じ `frame_log_rms`（n_fft = 8 x hop）。"""
    from .loudness import frame_log_rms
    lr = frame_log_rms(np.asarray(wav, dtype=np.float32).reshape(-1), hop=int(hop), n_fft=8 * int(hop))
    db = _TO_DB * np.asarray(lr, dtype=np.float64)
    return db[:n] if len(db) >= n else np.pad(db, (0, n - len(db)), mode="edge")


def gate_silence(f0: np.ndarray, uv: np.ndarray, wav: np.ndarray, *, hop: int,
                 ref_db: float, gap_db: float) -> tuple[np.ndarray, np.ndarray]:
    """`ref_db - gap_db` より小さいフレームを無声（f0 = 0、uv = 0）にする。ほかは触らない。"""
    f0 = np.asarray(f0, dtype=np.float32).copy()
    uv = np.asarray(uv, dtype=np.float32).copy()
    quiet = frame_db(wav, hop=hop, n=len(uv)) < float(ref_db) - float(gap_db)
    f0[quiet], uv[quiet] = 0.0, 0.0
    return f0, uv


def silence_gated(f0_extract: Callable, *, ref_db: float, gap_db: float) -> Callable:
    """`f0_extract(wav, sr, hop)` の結果に、入力全体の基準で無音ゲートを掛けた抽出器を返す。"""
    def f0x(wav, sr, hop):
        f0, uv = f0_extract(wav, sr, hop)
        return gate_silence(f0, uv, wav, hop=hop, ref_db=ref_db, gap_db=gap_db)
    return f0x


def silence_gate_manifest(gap_db: float | None) -> dict[str, Any]:
    """manifest に残す再現情報。`None` はゲートなし（旧い shard）。"""
    return {"f0_silence_gate": None if gap_db is None else {
        "reference": f"p{Q:g} of {int(WIN_SEC * 1000)} ms window RMS over the whole input (dBFS)",
        "frame": "frame_log_rms, n_fft = 8 x hop",
        "gap_db": float(gap_db)}}
