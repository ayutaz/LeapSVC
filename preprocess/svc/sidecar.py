"""WebDataset の sidecar JSON から、話者 id・曲名・帯域の足切りを決める。

[実行計画](../../doc/svc-plan.md) 12 節 P0-4。対象は `tts-dataset/japanese-singing-voice*`
の sidecar（`channel_id` / `channel_title` / `title` / `dataset_category` / `dataset_score` /
`duration_sec` / `language` などを持つ）。

**なぜ要るか。** `svc_dataset.py` は speaker id を**データディレクトリ名**から引き、
`_song_of()` は phrase 名から曲を引いて train/eval を曲単位で割ります。YouTube 由来の
素材は動画 1 本が 1 ファイルなので、**このままでは話者も曲も表現できません**。

**official を話者にしないこと。** 実測（2026-09-19）で `channel_title` に
`Warner Music Japan` が含まれていました。レーベルの公式チャンネルは複数アーティストを
抱えるので、channel = 話者になりません。`cover` は個人の歌い手チャンネルなので使えます。
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any

from leapsinger.config import MelSpec

from .audit import AuditThresholds

# `preprocess/svc/run.py` の `_SAFE` と同じ規約。**CJK は残す**（ASCII に削ると日本語題の
# 曲がすべて同じ名前へ潰れる）。
_SAFE = re.compile(r"[^\w-]+", re.UNICODE)

# 話者として扱ってよい `dataset_category`。**`official` を入れないこと**（上の注）。
SPEAKER_CATEGORIES = frozenset({"cover"})

# YouTube の題名に付く装飾。曲名を揃えるために落とす。**これは heuristic です** ――
# 完全な曲名抽出ではないので、結果は `duplicate_groups()` で目視できるようにしてあります。
_BRACKETS = re.compile(r"[【\[（(][^】\]）)]*[】\]）)]")
_MARKERS = re.compile(
    r"(歌ってみた|唄ってみた|歌わせて|カバー|cover(?:ed)?(?:\s+by)?|"
    r"music\s*video|official(?:\s+video)?|mv|full\s*ver\.?|short\s*ver\.?)",
    re.IGNORECASE,
)
# 区切り記号。最初の区切りより前を曲名とみなす。
_SPLIT = re.compile(r"[／/|｜~〜\-–—]")


def speaker_key(meta: Mapping[str, Any]) -> str | None:
    """話者 id（= データディレクトリ名）にできるなら返す。できなければ `None`。

    `None` を返した素材は**話者条件つきの学習に使えません**。捨てるか、ECAPA での
    話者クラスタリング（未実装）を通すかの判断が要ります。
    """
    if str(meta.get("dataset_category") or "") not in SPEAKER_CATEGORIES:
        return None
    channel = str(meta.get("channel_id") or "").strip()
    return channel or None


def fold_title(title: str) -> str:
    """題名を曲名へ畳む。`run.py` の `song_name()` と同じ正規化を最後に当てる。"""
    folded = _SAFE.sub("_", str(title).casefold()).strip("_")
    if not folded:
        raise ValueError(f"title folds to nothing: {title!r}")
    return folded


def song_key(meta: Mapping[str, Any]) -> str:
    """曲単位 split のための曲名を返す。

    **cover は同じ曲へ畳みます。** 別名のままにすると、同じ曲が train と eval に分かれて
    leakage します（`run.py` の `song_name()` と同じ理由）。**畳み方は heuristic** なので、
    採用する前に `duplicate_groups()` で目視すること。

    装飾を落とし切って空になる題名（`【歌ってみた】` など）は、**落とす前の形**を返します
    —— 空の曲名を作るよりは、畳めないまま残すほうが安全です。
    """
    title = str(meta.get("title") or "")
    if not title.strip():
        raise ValueError("title is empty")
    stripped = _MARKERS.sub(" ", _BRACKETS.sub(" ", title))
    head = _SPLIT.split(stripped)[0]
    for candidate in (head, stripped, title):
        try:
            return fold_title(candidate)
        except ValueError:
            continue
    raise ValueError(f"title folds to nothing: {title!r}")


def duplicate_groups(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[str]]:
    """同じ曲名へ畳まれた動画を返す（2 本以上のものだけ）。

    **除去はしません。** 同じ曲を別の歌手が歌ったものは学習素材として有効なので、
    捨てずに**同じ曲名を与えて split を揃える**のが正しい扱いです。ここが返すのは
    「畳み方が妥当か」を人が見るための材料です。
    """
    groups: dict[str, list[str]] = {}
    for row in rows:
        key = song_key(row)
        groups.setdefault(key, []).append(str(row.get("video_id") or ""))
    return {k: v for k, v in groups.items() if len(v) > 1}


def min_bandwidth_for_mel(mel: MelSpec) -> float:
    """帯域の足切りを **mel の上位 bin の中心**から決める。

    **sample rate から決めないこと。** 実測（2026-09-19）で MP3 由来の素材は実効帯域が
    p50 15,781 Hz で、`min_bandwidth_hz=16000` にすると **92% が `band_limited`** で
    弾かれます。しかし既定の mel（40–16,000 Hz / 128 bin）の**上位 bin の中心は 15,540 Hz**
    なので、**中心が実効帯域より上にある bin は 0/128** です。弾く理由がありません。
    """
    import librosa

    freqs = librosa.mel_frequencies(n_mels=int(mel.n_mels) + 2,
                                    fmin=float(mel.fmin), fmax=float(mel.fmax))
    return float(freqs[-2])


def audit_thresholds_for_mel(mel: MelSpec, **overrides: Any) -> AuditThresholds:
    """`min_bandwidth_hz` を mel から導いた `AuditThresholds` を作る。

    他の閾値は `AuditThresholds` の既定のままで、`overrides` で個別に上書きできます。
    """
    return replace(AuditThresholds(**overrides), min_bandwidth_hz=min_bandwidth_for_mel(mel))
