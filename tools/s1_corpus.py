#!/usr/bin/env python3
"""S1 の素材を用意する。**権利クリーンな素材だけ**を話者ごとのディレクトリへ並べる。

    uv run python tools/s1_corpus.py --out download/s1 --fetch      # 取得もする
    uv run python tools/s1_corpus.py --out download/s1              # 手元にある前提

[実行計画](../doc/svc-plan.md) **14 節 S1**。素材と除外の理由はそちら。

| 素材 | 話者 | 根拠（**許諾を待たない**） |
|---|---:|---|
| VocalSet（学習側 14 名） | 14 | **CC BY 4.0**（Zenodo 1442513） |
| NIT-SONG070-F001 | 1 | **CC BY 3.0**（DB 同梱の `data/COPYING`） |
| 波音リツ 3 音源 | 1 | サイト規約が商用可・音源の再配布可・クレジット不要 |

**評価に使う 3 曲は学習へ入れません。** 学習曲は上限との差 +1.0 点、未知曲は +12.4 点と
桁が違うので、混ざると比較が壊れます（`ritsu_soft` の `anywhere` のような別名版も落とす）。

**`--fetch` はインスタンス上で使うためのものです。** 手元から 5 GB を上げると実測で
**0.33 MB/s = 4 時間以上**かかりました。インスタンスから取り直すほうが速く、安いです。
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# [台帳](../doc/svc-dataset-ledger.md) の層別 hold-out（`split_by_group` の seed 42）。
# **評価側 6 名（female1 / female2 / female6 / male7 / male9 / male10）は学習に入れない。**
VOCALSET_TRAIN = ("female3", "female4", "female5", "female7", "female8", "female9",
                  "male1", "male2", "male3", "male4", "male5", "male6", "male8", "male11")
# 比較に使う 3 曲。casefold して部分一致で落とす（別名版も掴むため）。
EXCLUDED_SONG_KEYS = ("anywhere", "boukyakumoyou", "skyhighblue")

VOCALSET_URL = "https://zenodo.org/api/records/1442513/files/VocalSet11.zip/content"
NIT_URL = "https://hts.sp.nitech.ac.jp/archives/2.3/HTS-demo_NIT-SONG070-F001.tar.bz2"
NIT_SR = 48_000


def is_train_speaker(name: str) -> bool:
    """VocalSet の話者を学習に使うか。**評価側を混ぜない**ための唯一の判定。"""
    return str(name).strip().casefold() in {s.casefold() for s in VOCALSET_TRAIN}


def is_excluded_song(name: str) -> bool:
    """比較に使う曲なら True（学習へ入れない）。"""
    low = str(name).casefold()
    return any(k in low for k in EXCLUDED_SONG_KEYS)


def _run(cmd: list[str], **kw) -> None:
    print("  $", " ".join(cmd[:6]), "...", flush=True)
    subprocess.run(cmd, check=True, **kw)


def fetch(work: Path) -> None:
    """VocalSet / NIT-SONG070 / 波音リツ を取得する（インスタンス上で使う）。"""
    work.mkdir(parents=True, exist_ok=True)
    vz = work / "vocalset.zip"
    if not vz.exists():
        _run(["curl", "-sL", "-o", str(vz), VOCALSET_URL])
    nt = work / "nit.tar.bz2"
    if not nt.exists():
        _run(["curl", "-sL", "-o", str(nt), NIT_URL])
    if not (work / "HTS-demo_NIT-SONG070-F001").exists():
        _run(["tar", "-xjf", str(nt), "-C", str(work),
              "HTS-demo_NIT-SONG070-F001/data/raw"])
    for voice in ("kire", "normal", "soft"):
        code = ("import runpy,sys; sys.argv=['x','--voice','{v}']; "
                "runpy.run_path('preprocess/download_scripts/download_ritsu.py',"
                "run_name='__main__')").format(v=voice)
        _run([sys.executable, "-c", code], cwd=str(ROOT))


def build(out: Path, work: Path) -> dict:
    import numpy as np
    import soundfile as sf

    out.mkdir(parents=True, exist_ok=True)
    report: dict = {"speakers": {}, "excluded_songs": []}

    vz = next((p for p in (work / "vocalset.zip", ROOT / ".m0data/vocalset_audio.zip")
               if p.exists()), None)
    if vz is None:
        sys.exit("VocalSet の zip が見つかりません（--fetch を付けるか手元に置くこと）")
    z = zipfile.ZipFile(vz)
    pat = re.compile(r"/((?:male|female)\d+)/")
    n_v = 0
    for name in z.namelist():
        if not name.lower().endswith(".wav"):
            continue
        m = pat.search(name)
        if not m or not is_train_speaker(m.group(1)):
            continue
        rel = Path(name)
        dst = out / f"vocalset_{m.group(1)}" / (rel.parent.name or "song")
        dst.mkdir(parents=True, exist_ok=True)
        (dst / rel.name).write_bytes(z.read(name))
        n_v += 1
    report["speakers"]["vocalset"] = n_v
    print(f"VocalSet: {n_v} wav / {len(VOCALSET_TRAIN)} 話者", flush=True)

    raw_dir = next((p for p in (work / "HTS-demo_NIT-SONG070-F001/data/raw",
                                ROOT / ".m0data/p0/nit/HTS-demo_NIT-SONG070-F001/data/raw")
                    if p.exists()), None)
    n_n = 0
    if raw_dir is not None:
        for p in sorted(raw_dir.glob("*.raw")):
            data = np.frombuffer(p.read_bytes(), dtype="<i2").astype(np.float32) / 32768.0
            dst = out / "nit_song070" / p.stem
            dst.mkdir(parents=True, exist_ok=True)
            sf.write(str(dst / f"{p.stem}.wav"), data, NIT_SR, subtype="PCM_16")
            n_n += 1
    report["speakers"]["nit_song070"] = n_n
    print(f"NIT-SONG070: {n_n} wav", flush=True)

    n_r, n_ex = 0, 0
    for voice in ("ritsu", "ritsu_normal", "ritsu_soft"):
        base = ROOT / "download" / voice / "DATABASE"
        if not base.exists():
            continue
        for song_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            if is_excluded_song(song_dir.name):
                n_ex += 1
                report["excluded_songs"].append(f"{voice}/{song_dir.name}")
                continue
            dst = out / "ritsu" / f"{voice}_{song_dir.name}"
            dst.mkdir(parents=True, exist_ok=True)
            for w in song_dir.glob("*.wav"):
                shutil.copy2(w, dst / w.name)
                n_r += 1
    report["speakers"]["ritsu"] = n_r
    print(f"波音リツ: {n_r} wav / {n_ex} 曲を除外", flush=True)

    dirs = sorted(p.name for p in out.iterdir() if p.is_dir())
    report["n_speakers"] = len(dirs)
    report["speaker_dirs"] = dirs
    (out / "s1_corpus.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), "utf-8")
    print(f"話者ディレクトリ: {len(dirs)} -> {out}", flush=True)
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="話者ごとのディレクトリを作る先")
    ap.add_argument("--work", default="/root/s1_work", help="取得物の置き場")
    ap.add_argument("--fetch", action="store_true", help="素材の取得から行う")
    a = ap.parse_args()
    work = Path(a.work)
    if a.fetch:
        fetch(work)
    build(Path(a.out), work)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
