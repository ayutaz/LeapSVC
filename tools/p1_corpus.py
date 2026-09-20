#!/usr/bin/env python3
"""P1 の素材を用意する。`tts-dataset/japanese-singing-voice-vocal-only` の tar を
**インスタンス上で**流し読みし、`cover` だけを**話者ごとのディレクトリ**へ展開する。

    uv run python tools/p1_corpus.py --out download/tts_cover --hours 60 --seed 0
    uv run python tools/p1_corpus.py --out download/tts_cover --hours 60 --dry-run

[実行計画](../doc/svc-plan.md) **13 節 P1**。設計の根拠はそちら。要点だけ:

- **`cover` のみ。** `official` は channel = 話者になりません（`Warner Music Japan` が実在）。
  `diverse` も話者として扱いません（[商用リリースのデータ要件](../doc/commercial-release-data.md) 4.5 節）。
- **1 話者 1 曲が既定。** 実測で **1 channel あたり 1.1 曲**しか無く、目的は話者数を稼ぐこと
  なので、同じ channel から何曲も取りません。
- **手元に 200 GB を落とさないこと。** tar を**逐次読みしながら要る wav だけ書き出す**ので、
  ディスクには選んだぶんしか残りません。それでも**通信量は tar の全長**かかるため、
  **vast.ai のインスタンス上で実行します**（M3 で GTSinger を落としたのと同じ理由。
  通信は実費の 40% を占めました）。

**shard は約 1.6 GB / audio-hour** です。`--hours 60` で約 96 GB になるので、
インスタンスの disk を先に確認してください。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import tarfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from preprocess.svc.sidecar import speaker_key  # noqa: E402

REPO_ID = "tts-dataset/japanese-singing-voice-vocal-only"
N_SHARDS = 40


def plan_selection(entries: Iterable[Mapping[str, Any]], *, hours: float, seed: int,
                   per_speaker: int = 1) -> list[dict[str, Any]]:
    """取り込む entry を決める。**この関数だけが選定の真実**（CLI はこれを呼ぶだけ）。

    `random.Random` を使います（numpy の `Generator` と違いバージョン間でストリームが
    保証されるため、seed だけで再現できる）。`split.py` と同じ流儀です。
    """
    if float(hours) <= 0:
        raise ValueError(f"hours must be positive; got {hours}")
    budget = float(hours) * 3600.0

    usable = []
    for e in entries:
        if speaker_key(e) is None:          # cover 以外・channel 無しはここで落ちる
            continue
        sec = e.get("duration_sec")
        if not isinstance(sec, (int, float)) or sec <= 0:
            continue
        usable.append(dict(e))

    # 話者ごとにまとめてから、**話者の順序**をシャッフルする。曲ではなく話者を選ぶ形に
    # しないと、曲数の多い channel が先に埋めてしまい話者数が稼げない。
    by_spk: dict[str, list[dict[str, Any]]] = {}
    for e in usable:
        by_spk.setdefault(str(speaker_key(e)), []).append(e)
    for items in by_spk.values():
        items.sort(key=lambda x: str(x.get("video_id")))

    rng = random.Random(seed)
    speakers = sorted(by_spk)
    rng.shuffle(speakers)

    out: list[dict[str, Any]] = []
    total = 0.0
    for spk in speakers:
        for e in by_spk[spk][: max(1, int(per_speaker))]:
            sec = float(e["duration_sec"])
            if total + sec > budget:
                return out
            out.append(e)
            total += sec
    return out


def _shard_list(spec: str) -> list[int]:
    """`0-39` / `0,5,7` / 空文字（全部）を shard 番号の列に。"""
    if not spec:
        return list(range(N_SHARDS))
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return [i for i in out if 0 <= i < N_SHARDS]


def _stream(out_dir: Path, shards: Sequence[int], hours: float, seed: int,
            per_speaker: int, dry_run: bool) -> dict[str, Any]:
    """tar を逐次読みし、選んだ wav だけ `<out>/<channel_id>/<video_id>.wav` へ書く。

    **2 パスにしません。** sidecar JSON と wav が隣り合っているので、1 パスで
    「読みながら決める」形にします（2 パスにすると通信量が倍になる）。
    そのぶん選定は **channel 単位の到着順 + seed によるシャッフル**ではなく、
    **到着した channel を初出順に受け入れる**形になるので、`--seed` は
    **shard の処理順**に効かせます。
    """
    import requests
    from huggingface_hub import hf_hub_url
    from huggingface_hub.utils import build_hf_headers

    order = list(shards)
    random.Random(seed).shuffle(order)

    budget = float(hours) * 3600.0
    taken: dict[str, int] = {}
    total = 0.0
    picked: list[dict[str, Any]] = []
    seen = 0

    for idx in order:
        if total >= budget:
            break
        url = hf_hub_url(REPO_ID, f"data/vocals-{idx:04d}.tar", repo_type="dataset")
        with requests.get(url, headers=dict(build_hf_headers()), stream=True,
                          timeout=1800) as r:
            r.raise_for_status()
            with tarfile.open(fileobj=r.raw, mode="r|") as tf:   # ストリーム読み
                pending: dict[str, Any] | None = None
                for member in tf:
                    if not member.isfile():
                        continue
                    name = Path(member.name).name
                    if name.startswith("._"):
                        continue
                    if name.endswith(".json"):
                        pending = json.loads(tf.extractfile(member).read().decode("utf-8", "replace"))
                        seen += 1
                        continue
                    if not name.endswith(".wav") or pending is None:
                        continue
                    meta, pending = pending, None
                    spk = speaker_key(meta)
                    sec = meta.get("duration_sec")
                    if spk is None or not isinstance(sec, (int, float)) or sec <= 0:
                        continue
                    if taken.get(spk, 0) >= max(1, int(per_speaker)):
                        continue
                    if total + float(sec) > budget:
                        continue
                    taken[spk] = taken.get(spk, 0) + 1
                    total += float(sec)
                    picked.append({"video_id": meta.get("id"), "channel_id": spk,
                                   "channel_title": meta.get("channel_title"),
                                   "title": meta.get("title"), "duration_sec": sec,
                                   "dataset_score": meta.get("dataset_score"), "shard": idx})
                    if not dry_run:
                        dst = out_dir / spk
                        dst.mkdir(parents=True, exist_ok=True)
                        (dst / f"{meta.get('id')}.wav").write_bytes(tf.extractfile(member).read())
                    if total >= budget:
                        break
        print(f"  shard {idx:04d}: 選択 {len(picked)} / 走査 {seen} | 累計 {total / 3600:.2f} h",
              flush=True)

    return {"repo": REPO_ID, "shards": order, "seed": seed, "per_speaker": per_speaker,
            "hours_budget": hours, "hours_selected": round(total / 3600, 3),
            "n_speakers": len(taken), "n_clips": len(picked), "scanned_entries": seen,
            "dry_run": dry_run, "entries": picked}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="話者ごとのディレクトリを作る先")
    ap.add_argument("--hours", type=float, default=60.0, help="集める合計時間（13 節の決定は 50〜80）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-speaker", type=int, default=1,
                    help="1 話者から取る曲数。**既定 1**（実測で 1 channel 1.1 曲しか無い）")
    ap.add_argument("--shards", default="", help="例 0-9 / 0,5,7。既定は全部")
    ap.add_argument("--dry-run", action="store_true", help="書き出さずに選定だけ")
    ap.add_argument("--from-plan", default=None,
                    help="前回の `--dry-run` が書いた p1_corpus.json から**選び直す**。"
                         "`plan_selection()`（話者をシャッフルしてから選ぶ）を使うので、"
                         "到着順に依存しない再現可能な選定になる")
    a = ap.parse_args()

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if a.from_plan:
        prev = json.loads(Path(a.from_plan).read_text("utf-8"))
        chosen = plan_selection(prev.get("entries", []), hours=a.hours, seed=a.seed,
                                per_speaker=a.per_speaker)
        report = {"repo": REPO_ID, "from_plan": a.from_plan, "seed": a.seed,
                  "per_speaker": a.per_speaker, "hours_budget": a.hours,
                  "hours_selected": round(sum(float(e["duration_sec"]) for e in chosen) / 3600, 3),
                  "n_speakers": len({e["channel_id"] for e in chosen}),
                  "n_clips": len(chosen), "scanned_entries": len(prev.get("entries", [])),
                  "dry_run": True, "entries": chosen}
    else:
        report = _stream(out_dir, _shard_list(a.shards), a.hours, a.seed,
                         a.per_speaker, a.dry_run)
    (out_dir / "p1_corpus.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), "utf-8")
    print(f"話者 {report['n_speakers']} / clip {report['n_clips']} / "
          f"{report['hours_selected']} h -> {out_dir}")
    print(f"shard は約 1.6 GB/h なので、前処理後は約 "
          f"{report['hours_selected'] * 1.6:.0f} GB を見込む")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
