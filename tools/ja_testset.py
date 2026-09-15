#!/usr/bin/env python3
"""日本語の test set を作る（`natsume` / `oniku` の**学習に入っていない曲**から）。

    uv run python tools/ja_testset.py --audit out/ja_audit --speakers natsume oniku \\
      --seconds 20 --per-song 1 --out out/m5/testset_ja.json

**これは「未知話者」ではありません。** 手元の日本語 5 名は**全員 base の学習に入って
います**（`JA_Soprano_1` 12 / `JA_Tenor_1` 13 / `natsume` 20 / `oniku` 21 / `ritsu` 22）。
作れるのは **「未知曲・既知話者」** までで、記録にもそう残します（`kind: unseen_song`、
`seen_speaker: true`）。未知話者の日本語が要るなら東北きりたん / No.7 の取得が必要です。

**なぜ日本語が要るか。** CER（明瞭度）は言語に依存し、**いま読める日本語は hold-out
6 本＝target 本人の自己再構成だけ**です。話者類似度・timing・F0・V/UV は言語に依存しない
ので、VocalSet の既存の結果はそのまま使えます。

**選別の契約:**

- **学習に入った曲は絶対に混ぜない。** `ja_material_audit.py` の `free_files` だけを使い、
  `used_songs` に当たるものが紛れていたら**落とします**。
- **有声率 60% 以上**の区間を選ぶ（イントロ・間奏を掴まないため。M5 で実際に踏んだ）。
- **12 秒未満のクリップは作らない**（`speaker_similarity.py` の較正条件）。
"""
from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.ja_material_audit import song_of  # noqa: E402
from tools.m5_testset import (  # noqa: E402
    MIN_VOICED,
    measure_voiced_ratio,
    segment_holdout,
)

# `speaker_similarity.py` が 12 秒未満のクリップを拒否します（較正がその長さでしか通らない）。
MIN_CLIP_SECONDS = 12.0


def pool_from_audit(report: Mapping[str, Any], *, speaker: str, layout: str,
                    seconds: float,
                    durations: Mapping[str, float]) -> list[dict[str, Any]]:
    """棚卸しの結果から、test に使える曲の pool を作る。

    **`used_songs` に当たる曲が `free_files` に紛れていたら落とします。** 黙って通すと
    leakage し、しかも**指標は何も言わずに良く出ます**。
    """
    if speaker not in report:
        raise ValueError(f"{speaker!r} は棚卸しの結果にありません（{sorted(report)}）。"
                         "**base に入っていない話者なら、曲単位ではなく話者単位で"
                         "未知として扱えます**")
    if seconds < MIN_CLIP_SECONDS:
        raise ValueError(f"{seconds} 秒は短すぎます（{MIN_CLIP_SECONDS} 秒以上）。"
                         "話者類似度の較正がその長さでしか通りません")

    rep = report[speaker]
    used = set(rep["used_songs"])
    root = Path(rep["root"])
    pool = []
    for rel in rep["free_files"]:
        song = song_of(rel, layout=layout)
        if song in used:
            raise ValueError(
                f"{speaker}: {rel!r} の曲 {song!r} は学習に入っています。"
                "**棚卸しの結果が壊れているか、layout が合っていません**")
        dur = float(durations.get(rel, 0.0))
        if dur < seconds:
            continue
        pool.append({
            "speaker": speaker, "song": song, "clip": Path(rel).name,
            "path": str(root / rel), "seconds": float(seconds), "full_seconds": dur,
            # **「未知話者」と名乗らない。** 層を記録に残す。
            "kind": "unseen_song", "seen_speaker": True,
        })
    return pool


def main() -> int:
    import argparse
    import json

    import soundfile as sf

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit", required=True, help="ja_material_audit.py の出力ディレクトリ")
    ap.add_argument("--speakers", nargs="+", required=True)
    ap.add_argument("--layout", default="flat", help="素材の配置（UTAU DB は flat）")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--per-song", type=int, default=1,
                    help="1 曲から取る clip 数。**同じ曲から複数取ると独立ではありません**")
    ap.add_argument("--n", type=int, default=0, help="話者ごとの上限（0 なら全部）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-voiced", type=float, default=MIN_VOICED)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    clips: list[dict[str, Any]] = []
    for spk in a.speakers:
        f = Path(a.audit) / f"{spk}.json"
        if not f.exists():
            sys.exit(f"{f} がありません。先に ja_material_audit.py を回してください")
        rep = json.loads(f.read_text(encoding="utf-8"))
        root = Path(rep[spk]["root"])
        durs = {rel: float(sf.info(str(root / rel)).duration)
                for rel in rep[spk]["free_files"]}
        pool = pool_from_audit(rep, speaker=spk, layout=a.layout,
                               seconds=a.seconds, durations=durs)
        short = len(rep[spk]["free_files"]) - len(pool)
        print(f"[ja-testset] {spk}: 未使用 {len(rep[spk]['free_files'])} 曲 -> "
              f"{a.seconds:.0f} 秒に足りる {len(pool)} 曲（短くて落とした {short}）", flush=True)
        if not pool:
            continue
        got = segment_holdout(pool, n=(a.n or len(pool) * a.per_song),
                              seconds=a.seconds, seed=a.seed,
                              voiced_ratio=measure_voiced_ratio,
                              min_voiced=a.min_voiced)
        for i, c in enumerate(got):
            c["tag"] = f"ja_{spk}{i:02d}"
        clips.extend(got)
        vr = [c["voiced_ratio"] for c in got]
        print(f"  -> {len(got)} clip  有声率 {min(vr):.2f}〜{max(vr):.2f}", flush=True)

    out = {
        "clips": clips,
        "manifest": {
            "kind": "unseen_song",
            "seen_speaker": True,
            "note": ("**未知話者ではありません。** 手元の日本語 5 名は全員 base の学習に"
                     "入っています。未知話者の日本語が要るなら東北きりたん / No.7 の取得が"
                     "必要です"),
            "seconds": a.seconds, "per_song": a.per_song, "seed": a.seed,
            "min_voiced": a.min_voiced, "min_clip_seconds": MIN_CLIP_SECONDS,
            "speakers": list(a.speakers),
        },
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n合計 {len(clips)} clip -> {a.out}")
    print("  ** 層は「未知曲・既知話者」。**「未知話者」とは名乗らないこと** **")
    return 0


if __name__ == "__main__":
    sys.exit(main())
