#!/usr/bin/env python3
"""md のリンク切れとアンカー切れを検査する。

    uv run python tools/check_links.py

**アンカーまで見ます。** ファイルは在るのに見出しが無いリンクを実際に作りました
（日本語見出しの slug は記号の落ち方が直感と違います）。

**ネットワークは使いません。** `http` と `mailto:` は対象外です。到達性まで見ようと
すると CI がネットワークの調子で落ちるようになり、検査の意味が薄れます。

**slug の作り方は GitHub の規則そのものではありません。** 小文字化し、空白を `-` に
替え、バッククォートと記号を落とす、という近似です。**取りこぼすより余計に拾う側へ
寄せてあります**（見出しの実在だけを見たいので）。
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from pathlib import Path
from typing import Any

# 索引と規約が入っている場所。**skill も見ます** ―― skill 内のリンクも切れます。
DEFAULT_GLOBS: tuple[str, ...] = (
    "README.md", "README.en.md", "CLAUDE.md",
    "doc/*.md", ".claude/skills/*/SKILL.md",
)
_LINK = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")
_HEADING = re.compile(r"^#{1,6}\s+(.+)$", re.M)


def collect(root: Path | str, globs: tuple[str, ...] | list[str]) -> list[str]:
    """検査する md を root 相対のパスで返す（並びは安定）。"""
    root = Path(root)
    out: list[str] = []
    for g in globs:
        for p in sorted(glob.glob(str(root / g))):
            rel = os.path.relpath(p, root).replace(os.sep, "/")
            if rel not in out:
                out.append(rel)
    return out


def slug(heading: str) -> str:
    """見出しから anchor を作る（近似。docstring の注意を参照）。"""
    s = heading.strip().lower().replace("`", "").replace(" ", "-")
    return re.sub(r"[^\w-]", "", s)


def slugs_of(text: str) -> set[str]:
    return {slug(h) for h in _HEADING.findall(text)}


def broken_links(root: Path | str, files: list[str] | tuple[str, ...]) -> list[dict[str, Any]]:
    """切れているリンクを返す。空なら健全。

    返すのは `{"file", "target", "kind"}` で、`kind` は `file` か `anchor` です。
    **どちらなのかを分けます** ―― 直し方が違うためです。
    """
    root = Path(root)
    bad: list[dict[str, Any]] = []
    cache: dict[str, set[str]] = {}
    for rel in files:
        src = root / rel
        text = src.read_text(encoding="utf-8")
        for m in _LINK.finditer(text):
            target = m.group(2).strip()
            if target.startswith(("http://", "https://", "mailto:", "#!")):
                continue
            path, _, anchor = target.partition("#")
            tgt = (src.parent / path).resolve() if path else src.resolve()
            if path and not tgt.exists():
                bad.append({"file": rel, "target": target, "kind": "file"})
                continue
            if not anchor or not tgt.is_file():
                continue
            key = str(tgt)
            if key not in cache:
                cache[key] = slugs_of(tgt.read_text(encoding="utf-8"))
            if anchor not in cache[key]:
                bad.append({"file": rel, "target": target, "kind": "anchor"})
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description="md のリンク切れとアンカー切れを検査する")
    ap.add_argument("--root", default=".", help="repo の位置（既定は現在のディレクトリ）")
    ap.add_argument("--glob", action="append", default=None,
                    help="検査する md の glob（複数指定可。既定は README / CLAUDE.md / doc / skill）")
    a = ap.parse_args()

    root = Path(a.root)
    files = collect(root, tuple(a.glob) if a.glob else DEFAULT_GLOBS)
    if not files:
        print("検査する md がありません", file=sys.stderr)
        return 2
    bad = broken_links(root, files)
    print(f"md {len(files)} 件を検査しました")
    for b in bad:
        kind = "ファイルが無い" if b["kind"] == "file" else "見出しが無い"
        print(f"  {b['file']}: {b['target']}  <- {kind}")
    if bad:
        print(f"切れているリンク {len(bad)} 件", file=sys.stderr)
        return 1
    print("切れているリンクはありません")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
