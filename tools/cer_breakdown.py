"""CER を言語ごとに分けて読む（M5 ゴール 2 の明瞭度、内訳）。

**pooled の中央値は素材の混合を隠します。** M5 の公表値は「変換 19.7% / 上限 2.8% /
差 **+16.8 点**」でしたが、test set の 26 clip は **日本語 6 / ラテン語 10 /
イタリア語 5 / 英語 5** の混合で、差の中央値を言語別に割ると別物になります（実測）:

| 言語 | 本数 | 上限中央 | 変換中央 | 差の中央 |
|---|---:|---:|---:|---:|
| 日本語 | 5 | 2.4% | 7.1% | **+4.8 点** |
| 英語 | 6 | **0.0%** | 15.2% | **+13.1 点** |
| ラテン語 | 6 | **23.2%** | 52.3% | +20.5 点 |
| イタリア語 | 1 | 6.7% | 100.0% | +93.3 点 |

**上限が揃わない素材で差を出してはいけません。** 上限は「同じ内容を自分のボコーダーに
通した再合成」なので、書き起こしは source とほぼ一致するはずです。ラテン語は上限自身が
**23.2%** 外れており（`ドナのビスパーチェ` 対 `どなのびすパーチェ` ―― 同じ音の表記違い）、
**差は内容の劣化ではなく表記の揺れ**を拾っています。

`asr_cer.py` の `ceiling_unusable`（上限 CER > 0.5）は**clip ごとの安全弁**で、
「上限が 23% 揺れる素材」は通してしまいます。この文書はその**素材ごとの安全弁**です。
"""

from __future__ import annotations

import statistics as _stats
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# VocalSet の楽曲 -> 言語。**`straight` 等の技法名は楽曲ではありません。**
PIECE_LANG: dict[str, str] = {"dona": "la", "caro": "it", "row": "en"}

LANG_JA: dict[str, str] = {"ja": "日本語", "en": "英語", "it": "イタリア語", "la": "ラテン語"}

# 上限（GT mel -> 自分のボコーダー）の書き起こしが source とどれだけ揃えば
# 「その素材で CER を読める」とするか。実測で日本語 2.4% / 英語 0.0% に対し
# **ラテン語 23.2%** と大きく離れるので、閾値が微妙な判断をしているわけではない。
CEILING_STABLE_MAX = 0.10

# 中央値を 1 本や 2 本で出さない。**n=1 の +93.3 点を「イタリア語の劣化」と読まないため。**
MIN_CLIPS = 3


def clip_language(tag: str, *, piece: str | None) -> str:
    """clip の tag と楽曲名から言語を決める。

    **黙って既定値に落としません。** 知らない楽曲は例外にします ―― `ja` や `unknown` へ
    落とすと、**読めない素材が読めたことになります**。
    """
    if str(tag).startswith("holdout"):
        return "ja"
    if piece is None:
        raise ValueError(f"{tag}: 楽曲名がありません（holdout 以外は楽曲から言語を決めます）")
    key = str(piece).lower()
    if key not in PIECE_LANG:
        raise ValueError(f"{tag}: 知らない楽曲です（{piece!r}）。"
                         f"{sorted(PIECE_LANG)} のいずれか。**黙って既定値へ落としません**")
    return PIECE_LANG[key]


def piece_of(clip_name: str) -> str:
    """VocalSet のファイル名（`m10_dona_straight.wav`）から楽曲名を取る。"""
    parts = Path(str(clip_name)).stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"楽曲名が取れません（{clip_name!r}）。"
                         "`<歌手>_<楽曲>_<技法>.wav` を期待しています")
    return parts[1]


def _median(xs: Sequence[float]) -> float | None:
    xs = [float(x) for x in xs if x is not None]
    return float(_stats.median(xs)) if xs else None


def _group(clips: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ceil = _median([c["cer_ceiling"] for c in clips])
    conv = _median([c["cer_converted"] for c in clips])
    excess = _median([c["cer_excess_over_ceiling"] for c in clips])
    reason = None
    if len(clips) < MIN_CLIPS:
        reason = "too_few_clips"
    elif ceil is None or ceil > CEILING_STABLE_MAX:
        reason = "ceiling_unstable"
    readable = reason is None
    return {
        "n": len(clips),
        "ceiling_median": ceil,
        "converted_median": conv,
        # **読めない群では差を出しません。** 数字が残ると必ず引用されます。
        "excess_median": excess if readable else None,
        "readable": readable,
        "reason": reason,
        "ceiling_stable_max": CEILING_STABLE_MAX,
        "min_clips": MIN_CLIPS,
    }


def breakdown(clips: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """`lang` 付きの clip 記録を言語ごとに割る。

    `asr_cer.py` が既に落とした clip（`ceiling_unusable` / `asr_failed`）は数えません。
    `pooled` も返しますが、**言語が混ざっている限り `readable` は False** です ――
    混合の中央値を品質として報告させないためです。
    """
    use = [c for c in clips if not c.get("ceiling_unusable") and not c.get("asr_failed")]
    per: dict[str, Any] = {}
    for lg in sorted({str(c["lang"]) for c in use}):
        per[lg] = _group([c for c in use if str(c["lang"]) == lg])

    pooled = _group(use)
    if len(per) > 1:
        pooled["readable"] = False
        pooled["excess_median"] = None
        pooled["reason"] = "mixed_languages"
    pooled["reason_ja"] = ("言語が混ざった中央値は品質ではありません（内訳を読むこと）"
                           if len(per) > 1 else "単一言語")
    return {"per_language": per, "pooled": pooled,
            "n_used": len(use), "n_dropped": len(clips) - len(use)}


def label_clips(cer_clips: Sequence[dict[str, Any]],
                testset: dict[str, Any]) -> list[dict[str, Any]]:
    """`asr_cer.py` の clip 記録へ `lang` と `tag` を付ける。

    tag は `<曲>__<tag>` の後半、楽曲は `testset.json` の該当 entry から引きます。
    **index ではなく tag で引きます**（並びが変わると黙って別 clip の属性が付くため）。
    """
    piece_by_tag: dict[str, str | None] = {}
    attrs: dict[str, dict[str, Any]] = {}
    for kind in ("unseen", "holdout"):
        for i, e in enumerate(testset.get(kind, [])):
            tag = f"{kind}{i:02d}"
            piece_by_tag[tag] = None if kind == "holdout" else piece_of(e["clip"])
            attrs[tag] = {"gender": e.get("gender"), "speaker": e.get("speaker"),
                          "transpose": e.get("transpose"),
                          "voiced_ratio": e.get("voiced_ratio")}

    out = []
    for c in cer_clips:
        tag = str(c["file"]).split("__")[-1]
        if tag not in piece_by_tag:
            raise ValueError(f"{c['file']}: testset に {tag} がありません。"
                             "**黙って落とすと本数が減ったことに気づけません**")
        out.append({**c, "tag": tag, **attrs[tag],
                    "lang": clip_language(tag, piece=piece_by_tag[tag])})
    return out


def main() -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cer", required=True, help="`asr_cer.py` が書いた cer.json")
    ap.add_argument("--testset", required=True, help="out/m5/testset.json")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cer = json.loads(Path(a.cer).read_text(encoding="utf-8"))
    ts = json.loads(Path(a.testset).read_text(encoding="utf-8"))
    clips = label_clips(cer["clips"], ts)
    rep = breakdown(clips)

    print(f"=== CER の内訳（{a.cer}）===")
    s = cer.get("summary", {})
    if s.get("cer_excess_over_ceiling") is not None:
        print(f"公表値（pooled）: 変換 {s['cer_converted'] * 100:.1f}% / "
              f"上限 {s['cer_ceiling'] * 100:.1f}% / 差 {s['cer_excess_over_ceiling'] * 100:+.1f} 点"
              f"  <- **言語が混ざっているので品質として読まないこと**")
    print(f"使用 {rep['n_used']} / 除外 {rep['n_dropped']}")
    print()
    print("言語        本数   上限中央   変換中央   差の中央   判定")
    for lg, g in rep["per_language"].items():
        ex = f"{g['excess_median'] * 100:+6.1f} 点" if g["excess_median"] is not None else "   ――   "
        note = "読める" if g["readable"] else {
            "ceiling_unstable": f"上限が {g['ceiling_median'] * 100:.0f}% 揃わない",
            "too_few_clips": f"本数不足（{g['n']} < {MIN_CLIPS}）"}[g["reason"]]
        print(f"{LANG_JA.get(lg, lg):10s}  {g['n']:3d}    "
              f"{(g['ceiling_median'] or 0) * 100:6.1f}%   "
              f"{(g['converted_median'] or 0) * 100:6.1f}%   {ex}   {note}")

    read = {lg: g for lg, g in rep["per_language"].items() if g["readable"]}
    print()
    if read:
        print("**読める素材だけの差:** " + " / ".join(
            f"{LANG_JA.get(lg, lg)} {g['excess_median'] * 100:+.1f} 点（n={g['n']}）"
            for lg, g in read.items()))
    else:
        print("**読める素材がありません。** 差を報告できません")
    print("\n  ** 上限は「同じ内容を自分のボコーダーに通した再合成」。書き起こしが揃わない"
          "素材では、差は内容の劣化ではなく表記の揺れを拾う **")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({**rep, "clips": clips},
                                          ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
