#!/usr/bin/env python3
"""日本語素材のうち「base の学習に入っていない曲」を曲単位で出す。

    uv run python tools/ja_material_audit.py \\
      --manifests .m0data/m3/m3_out/manifests \\
      --root .m0data/gtsinger_ja/Japanese --speakers JA_Tenor_1 JA_Soprano_1

**なぜ要るか。** base の 23 話者には**日本語 5 名が全員入っています**
（`JA_Soprano_1` 12 / `JA_Tenor_1` 13 / `natsume` 20 / `oniku` 21 / `ritsu` 22）。
**未知話者の日本語は手元にありません**（東北きりたん・No.7 は未取得）。作れるのは
**「未知曲」**までなので、**どの曲が学習に入っていないか**を漏れなく出す必要があります。

**曲単位で判定します。** GTSinger のパスは `<技法>/<曲>/<Group>/NNNN.wav` で、
**同じ曲が複数の技法の下に出ます**。どれか 1 つでも学習に入っていれば、その曲は
test に使えません（このリポジトリの split は曲単位）。

**照合はファイル名ではなく相対パスで行います。** 連番は曲ごとに振り直されるので
（どの曲にも `0000.wav` がある）、名前で照合すると全部が「学習済み」に見えます。
`--verify-hash` を付けると、manifest の sha256 と手元の内容も突き合わせます。
"""
from __future__ import annotations

import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

# 曲名の表記ゆれを畳む。**`Heartful_Song` と `Heartful song` を別の曲にすると
# 曲単位 split が効きません**（CLAUDE.md の既知の落とし穴）。
_FOLD = re.compile(r"[\s_]+")


LAYOUTS = ("gtsinger", "flat")


def song_of(rel_path: str, *, layout: str = "gtsinger") -> str:
    """相対パスから曲名を取る（畳んだ形）。

    **`layout` を素材に合わせること。** 取り違えると曲単位 split が効かず leakage します。

    | layout | 想定する配置 | 曲 |
    |---|---|---|
    | `gtsinger` | `<技法>/<曲>/<Group>/NNNN.wav` | 2 番目の要素 |
    | `flat` | `wav/1.wav` / `<曲>/<曲>.wav` / `DATABASE/<曲>/<曲>.wav`（UTAU DB） | **1 ファイル = 1 曲**なのでファイルの stem |
    """
    if layout not in LAYOUTS:
        raise ValueError(f"知らない layout です（{layout!r}）。{list(LAYOUTS)} のいずれか。"
                         "**黙って既定に落とすと曲を取り違えて leakage します**")
    p = Path(str(rel_path).replace("\\", "/"))
    if layout == "flat":
        return _FOLD.sub(" ", p.stem).strip().casefold()
    if len(p.parts) < 2:
        raise ValueError(f"曲名が取れません（{rel_path!r}）。"
                         "`<技法>/<曲>/<Group>/NNNN.wav` を期待しています")
    return _FOLD.sub(" ", p.parts[1]).strip().casefold()


def _norm(p: str) -> str:
    return str(p).replace("\\", "/")


def audit(*, used: Iterable[str] | Mapping[str, str], disk: Sequence[str],
          disk_hashes: Mapping[str, str] | None = None,
          layout: str = "gtsinger") -> dict[str, Any]:
    """学習に入った相対パスと、手元の相対パスから、使える曲を出す。

    `used` は manifest の `sha256`（相対パス -> ハッシュ）でも、パスの列でも受けます。
    **手元に無い学習素材（`missing_from_disk`）も黙って捨てません** ―― 素材が欠けている
    ことに気づけなくなるためです。
    """
    used_hash = dict(used) if isinstance(used, Mapping) else {}
    used_paths = {_norm(p) for p in (used_hash or used)}
    disk_paths = [_norm(p) for p in disk]

    used_songs = {song_of(p, layout=layout) for p in used_paths}
    free = [p for p in disk_paths if song_of(p, layout=layout) not in used_songs]

    mismatch = []
    if disk_hashes:
        dh = {_norm(k): v for k, v in disk_hashes.items()}
        mismatch = sorted(p for p, h in used_hash.items()
                          if _norm(p) in dh and dh[_norm(p)] != h)

    return {
        "n_used_files": len(used_paths),
        "n_disk_files": len(disk_paths),
        "used_songs": sorted(used_songs),
        # **学習に 1 ファイルも入っていない曲だけ**が test に使えます。
        "free_songs": sorted({song_of(p, layout=layout) for p in free}),
        "free_files": sorted(free),
        "n_free_files": len(free),
        "missing_from_disk": sorted(used_paths - set(disk_paths)),
        "hash_mismatch": mismatch,
    }


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifests", required=True, help="base の manifest 群のディレクトリ")
    ap.add_argument("--root", required=True, help="手元の素材の親（歌手ディレクトリを含む）")
    ap.add_argument("--speakers", nargs="+", required=True)
    ap.add_argument("--layout", required=True, choices=list(LAYOUTS),
                    help="素材の配置。**取り違えると曲単位 split が効かず leakage します**")
    ap.add_argument("--verify-hash", action="store_true",
                    help="手元の内容が学習に使ったものと同じかを sha256 で確かめる（遅い）")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    report = {}
    for spk in a.speakers:
        mf = Path(a.manifests) / spk / "manifest.json"
        if not mf.exists():
            sys.exit(f"{mf} がありません。**この歌手が base に入っていないなら、"
                     "曲単位の判定ではなく話者単位で未知として扱えます**")
        used = json.loads(mf.read_text(encoding="utf-8"))["sha256"]
        # 手元は `JA_Tenor_1` を `JA-Tenor-1` で置くことがある（GTSinger の配布名）。
        cand = [Path(a.root) / spk, Path(a.root) / spk.replace("_", "-"),
                Path(a.root)]  # UTAU DB は root 直下が歌手ディレクトリのことがある
        root = next((c for c in cand if c.exists()), None)
        if root is None:
            sys.exit(f"手元に {spk} がありません（{[str(c) for c in cand]}）")
        disk = [str(p.relative_to(root)) for p in sorted(root.rglob("*.wav"))]
        hashes = None
        if a.verify_hash:
            print(f"[audit] {spk}: {len(disk)} ファイルを hash 中…", flush=True)
            hashes = {r: _sha256(root / r) for r in disk if _norm(r) in
                      {_norm(k) for k in used}}
        rep = audit(used=used, disk=disk, disk_hashes=hashes, layout=a.layout)
        report[spk] = {**rep, "root": str(root)}

        print(f"\n=== {spk} ===")
        print(f"  手元 {rep['n_disk_files']} ファイル / 学習に入った {rep['n_used_files']}")
        print(f"  曲: 学習に入った {len(rep['used_songs'])} / "
              f"**学習に入っていない {len(rep['free_songs'])}**")
        if rep["free_songs"]:
            print(f"  使える曲: {rep['free_songs']}")
            print(f"  使えるファイル: {rep['n_free_files']} 本")
        else:
            print("  ** 使える曲がありません（全曲が学習に入っています） **")
        if rep["missing_from_disk"]:
            print(f"  ** 手元に無い学習素材 {len(rep['missing_from_disk'])} 本 **"
                  f"（例: {rep['missing_from_disk'][:2]}）")
        if rep["hash_mismatch"]:
            print(f"  ** 内容が違う {len(rep['hash_mismatch'])} 本 **"
                  "（手元の素材が学習に使ったものと同じではありません）")
        elif a.verify_hash:
            print("  内容の一致を確認しました")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=1),
                               encoding="utf-8")
        print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
