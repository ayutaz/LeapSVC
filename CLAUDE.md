# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## このリポジトリについて

LeapSinger は歌声合成（SVS）用の音響モデルです。ランダムノイズではなく **F0 から作った「擬似 mel」（倍音インパルス＋白色ノイズ）を rectified flow の出発点 `x0`** にすることで、1 ステップ（`num_steps: 1`）で mel を生成します。mel → 波形は別リポジトリの NHVSing ボコーダー（`checkpoints/nhv_v3_1*.onnx`）が担当します。

現在のブランチ `feature/svc` では、既存 SVS を残したまま **歌声変換（SVC）経路**を追加中です。設計・調査ドキュメントは `doc/svc.md` が索引になっています（作業前に必ず読むこと）。

```text
SVS: 音素 + duration + F0        -> LeapSinger -> mel + F0 -> NHVSing -> WAV
SVC: source WAV -> content/F0/UV/loudness -> LeapSVC -> mel + F0 -> NHVSing -> WAV
```

## コマンド

**Python は `uv` 経由で実行します。**素の `python` / `pip`、および `uv pip` は使わないこと。依存の追加は必ず `uv add`、実行は必ず `uv run` です。

環境構築:

    uv sync --extra train --extra export --extra dev              # 依存を .venv へ（推論のみなら uv sync）
    uv sync --extra train --extra export --extra dev --extra eval # 評価も回すとき（speechbrain / torchaudio）
    uv add <package>                           # 依存を足すときは常に uv add（pyproject にも記録される）
    uv add --optional train <package>          # extra に足すとき（train / export）

- Python は **3.13 固定**です（`.python-version` と `pyproject.toml` の `requires-python = ">=3.13,<3.14"` の両方）。3.13 に狭めたことで解決が 1 本になり、lock は 3.13 用の最新（librosa 1.0.0 / numpy 2.5.2 / scipy 1.18.1 など）に揃います。別バージョンで動かす提案をするときは、まずこの固定を変える必要がある点を確認すること。
- `uv.lock` はコミット対象です。依存を変えたら lock の差分も一緒にコミットすること。
- **PyTorch は CUDA 版を pyproject が指定しています。** PyPI の Windows 版 torch は CPU ビルドなので、`[[tool.uv.index]] pytorch-cu130` + `[tool.uv.sources] torch` で PyTorch 公式 wheel index を明示しています。Windows / Linux は `uv sync` だけで GPU 版が入り、macOS は marker で PyPI（CPU/MPS）に落ちます。CUDA channel を変えるときは index の url（`cu126` / `cu128` / `cu130` / `cu132`）を書き換えて `uv lock` をやり直します。`uv add torch --index ...` を単発で打って pyproject の index 定義と食い違わせないこと。

前処理（SVS。データセット 1 つにつき recipe yaml が必要）:

    uv run python -m preprocess.run --recipe configs/recipes/<db>.yaml   # -> data/<db>/{shard.npz,metadata.json}

前処理（SVC。WAV ディレクトリから直接。音素ラベルは不要）:

    uv run python -m preprocess.svc.run --wav-dir download/ritsu --out data/ritsu_svc
    uv run python -m preprocess.svc.run --from-cache data/ritsu_svc/_cache --out data/ritsu_svc --subset-seed 1
    # 入れ子の深いコーパス（GTSinger の 1 歌手 = <技法>/<曲>/<Group>/NNNN.wav）
    uv run python -m preprocess.svc.run --wav-dir download/gtsinger/Japanese/JA-Soprano-1 \
      --out data/JA_Soprano_1 --song-parts 1 --max-hours 0.75

**2 段構成です。** 1 段目（重い・GPU）が ContentVec と RMVPE を回して `_cache/` へ、2 段目（軽い・CPU）が整列・正規化・次元削減を行って shard を書きます。`--from-cache` で 2 段目だけを回せるので、**補間方法や 256 次元 seed の ablation に ContentVec と RMVPE の再実行が要りません。**

**`--song-parts` を忘れないこと。** 既定は親ディレクトリ名を曲名にします。`<曲>/<曲>.wav` という配置ならそれで正しいのですが、GTSinger のように深いと**全部の曲が `Control_Group` に潰れます**。曲名は `_song_of()` の曲単位 split に使われるので、潰れると leakage します。`--max-hours` は歌手ごとの分量を曲をまたいで均等に選んで揃えるためのものです（base 事前学習では総時間より話者の多様性が効くため）。

**話者ごとにディレクトリを分けること。** `svc_dataset.py` は speaker id を**ディレクトリ名**から `spk_map` で引くので、1 つの shard に複数話者を混ぜると区別できません。M3 の素材一式は `tools/m3_corpus.py` が用意します。

学習（SVS も SVC も同じエントリポイント。`model.arch` で分岐）:

    uv run python -m train --config configs/3singer_ritsu3style_uv_gan2d.yaml \
      --data_dirs data/oniku data/natsume data/ritsu \
      --run_name <run> --out_root log --device cuda

    uv run python -m train --config configs/svc_base.yaml \
      --data_dirs data/target --run_name svc_target --out_root log --device cuda

    # target fine-tune（M4）。base を上書きしないよう run_name を必ず変える
    uv run python -m train --config configs/svc_target_ft.yaml \
      --data_dirs data/ritsu data/ritsu_normal data/ritsu_soft \
      --init_from ckpt_060000.pt --finetune \
      --run_name svc_ritsu_ft_01 --out_root log --device cuda

    # multi-singer base（M3）。spk_map と n_speakers は素材から生成した config に入る
    uv run python tools/m3_corpus.py --download --out data --write-config log/m3_base/config.yaml
    uv run python -m train --config log/m3_base/config.yaml --data_dirs data/<話者>... \
      --run_name m3_base --out_root log --device cuda

- `train.py` は step 100 / 1,000 / 10,000 で `[perf]` 行を出し、`log/<run>/perf.json` と TensorBoard の `perf/*` に step/s・examples/s・frames/s・peak VRAM を残します。vast.ai は時間課金なので、この値がそのまま料金の見積もりになります。
- 同じコマンドを再実行すると `log/<run_name>/ckpt_*.pt` の最新から**自動再開**します。別実験は必ず `--run_name` を変え、base checkpoint を上書きしないこと。
- `--init_from <ckpt> --finetune` は G 重みのみ読み込み、step / optimizer をリセット、D は新規です。**同一構造の checkpoint を前提**としており、SVS → SVC の部分ロードには対応していません（未実装）。

テスト:

    uv run python tools/smoke/run_smoke.py     # 全経路の疎通（合成音声・GPU で約3分）
    uv run python -m unittest test_svc_model -v
    uv run python -m unittest test_svc_preprocess -v  # SVC 特徴抽出前処理（重いモデル不要）
    uv run python -m unittest test_svc_dataset -v     # M0 の素材検査・split・音域集計
    uv run python -m unittest test_svc_metrics -v     # M5 の客観指標（timing / CER / 信号品質 / RTF）
    LEAPSINGER_INTEGRATION=1 uv run python -m unittest test_svc_preprocess_integration -v  # 実モデル（既定は skip）
    uv run python -m unittest test_svc_model.HarmonicSVCModelTests.test_forward_and_infer_reuse_flow_with_svc_conditioning
    uv run python tools/hooks/test_guard.py    # コマンド guard の回帰テスト（55 件）

    # 話者類似度。**encoder を替えたら必ず較正からやり直すこと**
    uv run python tools/speaker_calibrate.py --root .m0data/vocalset_calib \
      --encoder ecapa --glob "excerpts/straight/*.wav" --seconds 20 --device cpu
    uv run python tools/speaker_similarity.py --converted out/<変換結果> \
      --target download/ritsu --unrelated .m0data/unrelated_ref --seconds 20 --device cpu

    # M5 の客観指標。**上限（--self-check が出す `*_vocoder_only.wav`）を必ず併せて作ること**
    uv run python tools/timing_metrics.py  --dir out/<変換結果>
    uv run python tools/asr_cer.py         --dir out/<変換結果> --language ja --device cpu
    uv run python tools/signal_quality.py  --dir out/<変換結果> --device cpu
    uv run python tools/rtf.py --wav <vocal.wav> --ckpt <ckpt> --manifest <manifest> --device cpu
    uv run ruff check .                        # lint（`--fix` で自動修正）

`run_smoke.py` は 3rd-party API・学習・自動再開・推論・ボコーダー・前処理・ONNX 書き出しまでを 1 コマンドで通し、終了コードが失敗ステージ数になります。**依存やバージョンを変えた後、環境を移した後、学習を始める前に必ず走らせること。** 入力は合成波形なので品質の検証にはならず、配線が壊れていないことだけを示します。

単体テストは **493 件**（`test_svc_model` 58 / `test_svc_preprocess` 115 / `test_svc_dataset` 81 / `test_svc_metrics` 239）で、重いモデルもネットワークも使いません。`unittest discover` は hook で止めています（収集条件が暗黙で、走った件数が分かりにくいため）。上の 4 本を明示的に並べるか、`run_smoke.py` の `unittest` ステージを使ってください。後者は top-level の `test_*.py` を自動収集し、件数を表示します。`uv` を介さず素の Python で走らせると `librosa` 等が無く収集時に失敗します。

ONNX 書き出し（SVS のみ。実験的）。**OpenUTAU voicebank 書き出しは上流で削除されました**（2026-09-13 に取り込み。`export/dsconfig.py` と `export/openutau_assets.py` は存在しません）:

    uv run python -m export.cli --ckpt log/<run>/ckpt_050000.pt --out export/<name> \
      --model-name <name> --variant diffsinger --hop 512 --speaker embed

`infer.py` は CLI を持たないライブラリです（`load_acoustic` / `infer_mel` / `infer_svc_mel` / `load_vocoder` / `mel_to_wav`）。手元での再合成は `notebooks/resynth.ipynb`、書き出し確認は `notebooks/export_and_use_onnx.ipynb` を使います。

## アーキテクチャ

### 共有スタック（SVS / SVC 共通）

`HarmonicAcousticBase`（`leapsinger/models/acoustic_base.py`）が条件 `cond` を組み立て、`MelDilatedRectifiedFlow` が mel を直接 rectified flow で精製します。`HarmonicAcousticModel`（`acoustic.py`）が `_excitation_x0`（`harmonic_excitation.py`）と `_recon_loss` を足し、`HarmonicAcousticModelMultiSpk` が低次元話者ベクトル（`spk_bank` → `spk_proj` → cond 加算）を注入します。

損失は **flow 損失 + mel 再構成損失**が土台で、GAN（`gan.d_type: jcu | mel2d`）は `gan_start_step` 以降に後段で入れる二段構成です。GAN 側は `_forward_flow_gan` が `compute_loss` と数式・mask・RNG 順を一致させて再実装しているので、片方だけ変更すると学習が一致しなくなります。

### SVS と SVC の分岐点

分岐は **condition encoder だけ**です。それ以外（励起、flow backbone、損失、GAN、checkpoint 形式、NHVSing 互換 mel）は共有します。

| | SVS | SVC |
|---|---|---|
| 入力 | phoneme + duration + F0 | content + F0 + UV + loudness |
| encoder | `PhonemeEncoder` + `LengthRegulator` | `ContentAdapter`（`modules/encoders/content_adapter.py`） |
| モデル | `HarmonicAcousticModel(MultiSpk)` | `HarmonicSVCModel`（`models/svc.py`、`phoneme_encoder`/`length_regulator` を `None` にする） |
| dataset | `dataset.py` `LeapSingerDataset` / `shard.npz` | `svc_dataset.py` `SVCFeatureDataset` / `svc_shard.npz` |
| config | `model.arch` 未指定 | `model.arch: svc` |
| checkpoint `config.arch` | `harmonic` / `harmonic_multispk` | `harmonic_svc` |

`train.py` は `is_svc = cfg["model"].get("arch") == "svc"` で dataset・collate・eval・forward を切り替え、`infer.py` の `_build_from_config` は checkpoint に保存された `arch` でモデルを再構築します。**SVC を触るときも既存 SVS 経路（preprocess / 辞書 / export 契約）を壊さないこと。**

### config の構造

yaml は `mel` / `model` / `excitation` / `train` / `gan` / `data` の 6 セクションです。`mel` は `MelSpec`（`leapsinger/config.py`）として前処理・loader・励起 hop で共有され、常に一致している必要があります（44.1 kHz / hop 256 / n_fft 2048 / 128 mel / 40–16000 Hz、NHVSing V3 互換）。

`configs/.gitignore` は top-level の `configs/*.yaml` を無視し、`3speaker_gan2d.yaml` / `3singer_ritsu3style_uv_gan2d.yaml` / `svc_base.yaml` / `svc_base_multi.yaml` / `svc_target_ft.yaml` / `svc_target_ft_gan.yaml` の 6 つだけを公開対象にしています。新しい config を追加してもコミット対象にならない点に注意。

`svc_target_ft.yaml`（M4 の target fine-tune）は **header に checkpoint 選択規則を書いてから**走らせた config です。実験の後に規則を決めると train loss で選んでしまうので、この順序自体が成果物の一部です。

### SVC のデータ契約（厳格）

```text
data/<db>/metadata.json     # {"content_dim": 256, "frame_rate": 172.265625, "phrases": {"<name>": <frames>}}
data/<db>/svc_shard.npz     # <name>|content [T,C] / |f0_interp [T] / |uv [T] / |loudness [T] / |mel [128,T]
```

**すべての `T` は完全一致させます。** loader（`svc_dataset.py`）は暗黙の transpose や補間を行わず、幅・フレーム数の不一致を例外にします。この「黙って直さない」性質は前処理ミスを早期に露出させるための設計なので、緩めないこと。

**`content_dim` は 256 です。** `doc/svc-content-encoder.md` の決定により、ContentVec 768 次元から固定ランダムに選んだ **256 次元**を shard に書きます（生の 768 は 1 段目の cache に残るので、部分集合を変える ablation は 2 段目の再実行だけで回せます）。**loader に切り出しをさせません** — 「黙って直さない」契約を崩すためです。`configs/svc_base.yaml` も `content_dim: 256` です。

**WAV から shard を生成する経路は実装済みです。** `preprocess/svc/` の構成:

| モジュール | 役割 |
|---|---|
| `align.py` | SSL 50 Hz -> mel grid（172.265625 Hz）の **left（直前保持）**整列。比が整数にならないので方式が契約になる |
| `subset.py` | ContentVec 768 -> **256 次元**の部分集合（seed 0 が既定。index を manifest へ） |
| `loudness.py` | フレーム log-RMS（**mel とフレーム数が一致**）と dataset 統計での正規化 |
| `audit.py` / `coverage.py` / `split.py` / `report.py` | M0 の素材検査・音域と技法の集計・group 単位 split |
| `chunk.py` | 長い曲を固定長の phrase へ切る。**有声率が `--min-voiced` 未満の chunk は捨てる**（イントロが丸ごと無声になるため。実測では 89 chunk 中 39 件が除外され、うち 35 件は完全に無声だった） |
| `extract.py` / `encoders.py` | 1 段目。ContentVec と RMVPE を**引数で受け取り**、`{content, f0_hz, uv, loudness, mel}` を返す |
| `shard.py` | 2 段目。整列・部分集合・正規化を当てて `svc_shard.npz` を書く。`features_to_item()` は推論側に同じ正規化を当てる |
| `run.py` | CLI。`--from-cache` で 2 段目だけ再実行できる |

**確認済み:** WAV から shard までコマンド 1 本で作れ、再実行で **bit 一致**します（`np.savez` は zip にタイムスタンプを埋めるので自前で決定的に書いています）。

manifest には encoder の model revision と層、sample rate、hop、**SSL の stride と補間方法**、正規化、**256 次元の index と seed**、loudness の定義、F0 extractor version、入力 WAV の checksum を記録します。

## 自動化と安全装置

`.claude/` にこのリポジトリ用の skill と hook を置いています。

| 種類 | 名前 | 役割 |
|---|---|---|
| skill | `leapsinger-tdd` | このリポジトリでの TDD の当て方（重いモデルの扱い、契約テスト、置き場） |
| skill | `leapsinger-verify` | 依存・環境を変えた後の疎通確認の回し方と結果の読み方 |
| skill | `leapsinger-experiment` | 学習実験を事故なく回す手順（run 名、無視される設定、記録、主張の範囲） |
| skill | `leapsinger-docs` | 確度ラベルと主張規則を保ったままドキュメントを更新する作法 |
| skill | `vast-instance` | vast.ai インスタンスの検索・作成・回収・破棄、実運用で踏んだ落とし穴 |
| hook | `tools/hooks/guard_commands.py` | `PreToolUse` で「常に間違い」なコマンドを実行前に止める（回帰テスト 55 件） |

hook が止めるもの: `uv pip` / 素の `pip` / 素の `python`（**`-m` と `.py` だけでなく `-c` と `-`（stdin）も**。この 2 つは素通りしていました）、**rebase / merge の途中での `uv run` / `uv sync` / `uv add`**（作業ツリーが過去のコミットなので、その時点の `pyproject.toml` で環境が再同期され、生成された `uv.lock` が rebase を止めます。実際に torch が 2.14 → 2.13 に入れ替わって中断しました）、`.env` の staging、`git push --force`、`git reset --hard`、`log|data|checkpoints|.git` の `rm -rf`、`vastai` の直接叩き（料金確認を飛ばすため）、`unittest discover`、**手元での学習**（device によらず）、**既存 ckpt がある run へ `--init_from` を渡す**こと（`train.py` はこれを黙って無視して自動再開します）、**取得スクリプトの `-m` 実行**（`_gdrive` の兄弟 import が解決できず必ず失敗）、**`CUDA_VISIBLE_DEVICES=""`**（空文字は未設定扱いで CUDA が隠れない。`-1` が要る）、**角括弧で自己一致を外していない `pkill -f`**（このハーネスは `bash -c` で走るのでシェル自身に一致し、後続のコマンドごと落ちます）。

止めすぎると自動運転が壊れるので、判断の余地がないものだけを対象にしています。どうしても必要なときはコマンド末尾に `# guard:allow` を付けると通ります。ルールを足したら `tools/hooks/test_guard.py` にケースも足してください。

**誤検知が 1 つ分かっています。** guard はコマンド文字列を見るので、**ドキュメントの本文に `train.py` や学習コマンド例を書き込む**とき（heredoc で md を編集するなど）にも学習の起動と見なして止まります。この場合は `# guard:allow` を付けてください。

**ヒアドキュメント経由ではバックスラッシュが 1 段外れます。** `<<'PY'` で quote していても、Python には `\\n` が `\n` として届きます。md のコードブロックに行継続の `\` を書くと**行が連結されて壊れます**（実際に起きました）。`chr(92)` で組み立てるか、書いた後に必ず読み返して確認すること。

## この開発での約束事

**実装はすべて TDD で行います。** 失敗するテストを先に書き、失敗を確認し、通す最小限のコードを書く。先に書いたテストが無い実装コードは破棄してやり直します。規律は `superpowers:test-driven-development`、このリポジトリ固有の当て方（重い事前学習モデルをどう避けるか、何を契約テストにするか）は `leapsinger-tdd` skill を参照してください。

`doc/` 配下の文書は確度ラベル（**確認済み / 決定 / 推奨 / 見積もり / 仮説 / 未実装 / 要ユーザー判断**）で記述を区別しています。ドキュメントを更新するときはこのラベル体系を維持してください。特に `doc/svc-prior-art-license.md` の主張ルールに従います。

- 「確認済み」はコード・実行 artifact・一次資料のいずれかを示せる場合のみ。
- 「Seed-VC より良い」は同一 test set の blind comparison 後にのみ使う。
- 「リアルタイム」は対象ハードウェアでの end-to-end latency 実測と連続動作後にのみ使う。
- 「世界初」「唯一」は使わない（rectified-flow SVC も harmonic modelling も先行研究がある）。
- 「1-step」は acoustic flow の step 数であり、pipeline 全体の話ではない。

現在の到達点は**完了レベル 4（品質比較）**です。実音声から shard を作り、23 話者・約 18 時間の multi-singer base を **60,000 step** 学習し、そこから波音リツへ **20,000 step の fine-tune** まで実施しました（M0〜M4 完了）。**M5（Seed-VC 比較）は完了**しました。**話者類似度はほぼ同等**（訂正後 0.5899 対 0.5912。旧 0.4910 / 0.4981 は上限混入による測定誤り）ですが、**blind preference で負けた**ため（25 判定中 21）、事前登録した規則により**「Seed-VC より良い」とは書けません**。**SVC 推論の既定は 16 step です**（`tools/svc_defaults.py`。掃引で決定）。**blind listening test も実施済み**です（2026-09-13、26 ペア、N=1 非公式）。**streaming student（M6）は未着手**です（`doc/svc-implementation-status.md` の検証済み / 未検証の境界を参照）。

**M4 で分かった trade-off（実測）:** fine-tune を進めるほど **target らしさは上がり**（話者類似度の回復率 45.1% → 58.0%、自己再構成は上限比 94.8% → 98.1%）、**未知 source の内容保持は単調に落ちます**（cos 0.8599 → 0.8359）。config に事前登録した規則（未知 source の cos が base から 0.02 を超えて落ちた checkpoint は選ばない）で **`ckpt_010000` を選択**しました。train loss だけで選ぶと 20,000 step を選んでしまいます。

**話者類似度は encoder とクリップ長の両方に依存します（実測）。** `transformers` の x-vector は歌声の同性ペアを分離できず（重なり 83.3%、12 秒にしても 77.0%）、**ECAPA-TDNN を 12 秒以上**で使って初めて事前登録した合格条件（20% 以下）を満たします（19.8%、20 秒で 17.3%）。`tools/speaker_similarity.py` は **12 秒未満のクリップを拒否**します。一度「測れない」と結論しましたが、原因の半分は 6 秒に切って測っていたことでした。**encoder を替えたら `tools/speaker_calibrate.py` で必ず較正をやり直すこと。**

## 既知の落とし穴

- `train.py` は **gradient accumulation を実装していません**。config に `accum_steps: 2` があっても無視されます（互換のために残されている値）。実効 batch を増やす提案をする際はこの前提を確認すること。
- **ローカルの GPU は `nvidia-smi` が正常に見えても context 生成に失敗することがあります**（`CUDA error: CUDA-capable device(s) is/are busy or unavailable`）。`torch.cuda.is_available()` は driver の有無しか見ないので **True を返しても使えるとは限りません**。判定するなら `torch.zeros(1, device="cuda")` を実際に確保すること。この状態では推論スクリプトも落ちるので `--device cpu` で回します。
- **CPU で回すときは `CUDA_VISIBLE_DEVICES=-1` を渡すこと。** torch 2.13 の optimizer は `step()` ごとに `torch.accelerator.current_stream()` を呼ぶため、CPU tensor しか無くても壊れた CUDA に触って落ちます。**`""` では効かず `-1` が要ります。** `run_smoke.py --device cpu` は自動で渡します。`train.py` の `pin_memory` も `_loader_kwargs()` で CUDA のときだけ有効です（回帰テストは `test_svc_model.LoaderKwargsTests`）。
- **`tools/smoke/` の合成データは `configs/svc_base.yaml` の `model.content_dim` を読んで作ります。** ここを定数に戻すと、config を変えたときに SVC の学習・再開・推論ステージが黙って落ちます（実際に起きました）。
- **出力のスペクトル傾斜は入力の F0 に強く従います。** 40 clip の実測で、明るさが同じ男女の歌唱を**同じ target** へ変換すると 540 Hz 対 1067 Hz になりました（target を男性に替えても男性 source は動きません）。content と loudness を固定して F0 だけを ±12 半音した交差実験で確定しています（男性 +12 で 540 → **1196 Hz**、女性 −12 で 1067 → 378 Hz）。**低い声の source には `--transpose` に +7 を渡すこと**（`tools/svc_defaults.py` の `SVC_TRANSPOSE_LOW_VOICE`。2026-09-16 の掃引で決定）。~~+7〜+12~~ という範囲での推奨を、**日本語 8 clip の掃引で +7 に確定**しました ―― **それまでの +12 は両方の軸で劣ります**（回復率 90.8% → 81.6%、CER の上限との差 +12.5 → **+27.0 点**で制約 17.4 点を外れる）。**理論値の F0 一致（+9.7 半音）より少ない移調のほうが良い**ので、**話者類似度は単純な F0 の一致では決まっていません**。**n=8・1 話者・1 target なので、別の声では測り直すこと。** 移調なしの数値だけでモデルの良し悪しを判断しないこと。
- **明るさを符号つき平均で評価しないこと。** 上限（GT mel の再合成）より明るい clip と暗い clip が打ち消し合い、平均は良く見えるのに実際は両方向へ外れている、ということが起きます（実測で範囲 −70% 〜 +33%）。**ceiling からの距離（絶対値）**で見ます。この誤りで「多 step にすると明るさが戻る」という結論を一度出しました。
- **`num_steps` は「明るさでは同等、細部と話者性では明確に効く」。** 60,000 step の base で
1 step と 16 step の**明るさ**は実質同等ですが（|上限比| 34.3 対 34.0）、**細部と話者性は
16 step が明確に上**です（細部比 0.755 → 0.957、話者性の回復率 78.2% → 90.6%。自己再構成）。
**未知 source でも 58.8% → 69.4%** で、内容 cos は 0.868 → 0.852 と少し落ちます。
**費用はほぼ無視できます**（flow の RTF 0.044 → 0.065。step 16 倍でも 1.5 倍にしかならない
のは大半が呼び出しのオーバーヘッドだから）。**「1 step で十分」を細部や話者性へ広げないこと。**
比較や報告では **step 数を必ず併記**すること。
- **過平滑には GAN が効きます（2026-09-02 実測）。** `configs/svc_target_ft_gan.yaml` で
fine-tune すると、未知 source の回復率が 69.4% → **75.3%**、signal quality も上限との差が
−0.019 → −0.008 へ改善します。**trade-off は出ません**（内容 cos は微増、F0 相関は一定）。
**改善の大半は最初の 2,500 step**で、以降 20,000 step までほぼ平坦です。**細部比は単調に
増え続けますが（1.08 → 1.46）、話者性は追随しません** — 細部の量は目的ではありません。
- **話者性が弱い原因は過平滑です（実測で切り分け済み）。** ボコーダーと mel 表現は無罪
（自己再構成で 96.4% 保つ）、source 話者の漏れも無い（下限に張り付く）、話者条件も効いている
（target 指定で 0.24 → 0.52）。**予測 mel の細部が 25% 欠けている**のが原因で、
**base も fine-tune も `gan.enabled: false`**（flow 損失 + 再構成損失のみ）でした。
step を増やすと細部と話者性が**同時に単調に**戻ることで因果を確認しています。
- **rebase の途中で `uv run` を打たないこと。** 作業ツリーが過去のコミットにあるので、
**その時点の `pyproject.toml` で環境が再同期**されます（実測で torch 2.14 → 2.13）。生成された
`uv.lock` が未追跡ファイルとして残り、**rebase が "Please move or remove them" で中断**しました。
競合の解決は `sed` / `awk` / エディタで行い、**テストと lint は rebase を終えてから**回します。
hook が止めます。
- **出力の明るさを上げて「良く聴こえる」ようにしないこと（2026-09-13 決定）。**
target 本人の録音は centroid 1160 Hz、LeapSVC は 1067 Hz でほぼ一致します。Seed-VC は
2149 Hz で **1.85 倍明るく**、そのぶん大きく・近く聴こえて preference を取ります。
**明るくすれば選好は取れますが target から離れます。** 目標は本人に似ていることなので、
モデル側では足しません（聴感上の大きさはミックス段の話）。**レベルを上げても直りません** ―
原因は明るさであってレベルではないためです。
- **A/B 比較で RMS を揃えても、明るさが揃っていなければ公平ではありません。**
実測で Seed-VC は 26 clip すべて LeapSVC より明るく（centroid 比 中央値 1.80）、
評価者は「選んだほうが大きかった」と述べました。**明るい音は同じ RMS でも大きく・
近く聴こえます**（LUFS の差は中央値 −0.26 LU しかなく、これだけでは説明が付きません）。
**唯一の引き分けは明るさの差が最小の clip**でした。
- **選択肢に無い defect は、近いラベルに化けて記録されます。** 「音量が小さい」を
用意していなかったため、評価者は「音量が揺れる」を選び、私はそれを速い変動として
測って空振りしました。**名付けの選択肢は、想定していない答えを書ける形にすること。**
- **聴こえた defect が客観指標で再現しないことがあります。** 聴取で 6 本中 5 本が
「音量が揺れる」と答えたので `tools/loudness_stability.py` を作りましたが、**LeapSVC の
ほうが source の抑揚を忠実に追っていました**（残差 2.12 対 4.67 dB、速い成分でも 1.49 対
3.58）。**ここで指標を取り替え続けると、結果に合う指標を選ぶことになります。** 否定的な
結果をそのまま残し、次は評価者に聞き直すこと。
- **clip ごとの移調を渡さないと、意図した移調が「音程の誤り」になります。**
`tools/pitch_metrics.py` は `--testset` が無いと**黙って移調 0 と仮定**します。男声 source →
女声 target の **+12** をそのまま誤差として計上し、26 clip 中 11 本が 12 半音ずれていると
記録されました。**集計が中央値なので headline には出ません**（過半数が正常なら中央値は
無傷）。`resolve_transposes()` が `*_convert.json` から拾います。
- **上限（`*_vocoder_only.wav`）を変換結果として数えないこと。** `--self-check` を
付けて変換すると 1 clip につき `_converted` / `_source` / `_vocoder_only` が並びます。
`tools/speaker_similarity.py` は `_source` しか除いておらず、**変換結果 13 本と上限 13 本を
混ぜて**平均していました。**baseline のディレクトリには上限が無い**ので、別種のファイルを
比べることになります（実測で話者類似度が 0.5886 → 0.4981 に見えていました）。
`pick_clips()` が両方を落とします。**`--n-clips` の既定は 16 で、26 clip を黙って切ります** ―
切り詰めは report に残るようにしました。
- **fine-tune は明瞭度を改善します（2026-09-17 実測）。壊してはいません。** hold-out 6 clip
（リツの未知曲）で上限との差は **base 60,000 が +25.5 点、非 GAN ft が +13.2 点、GAN ft が
+12.7 点**。**話者類似度も 77.9% → 91.4% と同じ向き**で、**trade-off は出ません**。
**非 GAN と GAN がほぼ同じ**（+13.2 対 +12.7）なので、**効いているのは target 適応であって
GAN ではありません**。M4 の「未知 source の内容保持が落ちる」と矛盾しません ――
**target の未知曲**への汎化と**未知話者**への汎化は別物です。
- **推論時は checkpoint ごとに学習時の manifest を使うこと。** 正規化統計が run ごとに違います
（実測で ritsu の `loudness_mean` が −4.6289 / −4.6554 / −4.7417）。**揃えると別の実験に
なります。**
- **事前登録する閾値は、ノイズを実測してから立てること（2026-09-17 の反省）。**
checkpoint ごとの明瞭度で「2 点以上良くなったら既定を変える」と登録しましたが、**実測の
振れ幅は 1 clip あたり最大 73.6 点**でした（隣接 checkpoint 間）。**中央値では 2.3 点の
「改善」が出るのに、clip 単位では 2 勝 2 敗 2 分（p = 1.0）**です。**中央値だけを見る規則は
この壊れ方を通します** ―― 規則に**対応のある検定**を入れるか、**閾値をノイズの実測に対して**
立てること。
- **明瞭度は fine-tune の checkpoint に依存しません（2026-09-17 実測）。** 2,500〜15,000 step で
上限との差は +10.4〜+20.2 点と**単調でなく**、clip 単位では差が検出できません。話者類似度も
89.0〜92.1% でほぼ平坦です。**したがって学習曲 +1.0 点 対 未知曲 +12.4 点という汎化の差は、
fine-tune の過学習では説明できません** ―― 原因は **base の汎化**か課題の本質的な難しさです。
- **明瞭度の劣化は汎化の問題です（2026-09-16 実測）。** 学習曲の自己再構成は上限比
**+1.0 点**、hold-out は **+12.4 点**、他話者からの変換は **+12.5 点**。**学習曲ではほぼ
完全に再構成できます。** したがって **content 表現（256 次元）・整列・flow の容量は無罪**で、
**話者転移も無罪**です（hold-out と他話者で差が無い）。残る候補は**過学習**です
（fine-tune はリツ 7.6 時間で 10,000〜15,000 step）。**「content_dim を増やす」という筋は
この測定で否定されました。**
- **長く走るコマンドを `| head` や `| tail` に通さないこと。** `head` がパイプを閉じると
**本体が途中で死にます**。実測で 8 本の変換が **6 本で止まり**、最後に書くはずだった
`clips.json` も残りませんでした。しかも**終了コードは 0** です。`run_in_background` で
出力をファイルへ落とし、**必要な行は後から読むこと**。`| tail` は全部を待つので殺しませんが、
**完了まで 1 行も見えません**（これも実際に踏みました）。
- **学習 hold-out は後から再現できます（2026-09-16 確認）。** `svc_dataset.py` の split は
`random.Random(42).sample(sorted(曲名), eval_songs)` で、曲名は `preprocess.svc.run` が
**casefold してから** `[^\w-]+` を `_` に置き換えた形です。**casefold を忘れると並び順が
変わり、別の 3 曲が hold-out になります**（実際に一度取り違えました）。正しく再現すると
M5 の hold-out 3 曲（`anywhere-3_normal` / `boukyakumoyou-3_normal` / `skyhighblue-3_normal`）
と一致し、**M5 の hold-out が本物の学習 hold-out であることも確認できます**。
**どの曲が学習に入っていたかを後から判定できる**ので、「容量の問題か汎化の問題か」の
切り分けに使えます。
- **`--out` の親ディレクトリを作らないツールを増やさないこと。** `speaker_similarity.py`
だけが作っておらず、**ECAPA を 4 条件ぶん CPU で回し切った後に `FileNotFoundError` で
結果が丸ごと失われました**。しかも**シェルの `for` ループは終了コード 0 を返した**ので、
成功したように見えました。**重い測定ほど、結果を捨てる失敗が痛いです。**
`test_svc_model.py` の `test_every_tool_creates_its_output_directory` が `tools/*.py` を
走査して防ぎます。
- **主観テストの anchor は `tools/blind_test.py prepare --anchor-target ... --anchor-foil ...`で入れます（2026-09-15 実装）。** 正解は `anchors.json` にだけ置き、**ページと `key.json`
には出しません**。`tally` が別に採点し、**合格線 75% を下回ると「本番の拮抗を『2 系が同一』と
読まないこと」と警告**します。**引き分けは不正解**（target 本人 対 無関係な話者で引き分けなら
判別できていない印）、**未記入は不正解にしません**。**anchor だけ両側を同じ sample rate へ
対称に落とします** ―― 素材が別コーパスから来るため（実測でリツ 44.1k / 棗 48k / 鬼灯 96k）で、
**片側だけ触ると帯域が手がかりになります**。**A / B の系ペアは揃えず例外のまま**です。
- **主観テストに catch trial（anchor）を入れること。** 入れないと**「2 系が同一」と
「評価者が課題を遂行できていない」を区別できません**。target 類似の blind で実際に踏みました
（判定 4 票が**すべて「後に聴いた側」**で、評価者は「意味がないように思えます」と述べた）。
**target 本人の録音 対 無関係な話者**のペアを混ぜておけば、判別できる耳であることを先に
確認できます。**A → B の連結再生は構造的に B を最後に置く**ので、判別が効かない課題では
recency が空白を埋めます。
- **集計は side の偏りも出すこと。** `tools/blind_test.py` の `tally()` は `sides` と
`p_side` を返し、**位置の偏りが系の差より強いときに警告します**。実測で preference は
A 10 / B 15（p_side 0.4244、系は 0.0009）で無罪、similarity は **A 0 / B 4** でした。
**p_side を見ずに系の勝敗を読むと、位置の偏りを系の差と取り違えます。**
- **客観で同等と分かっている軸を、主観で測ろうとしないこと。** 話者類似度 0.5899 対 0.5912
（相対差 **0.2%**）は、**人間が判別できないことを予告していた**と読めます。「測れていない」より
先に「**差が無い**」を考えること。
- **「測ってあるから聴かなくてよい」と観測を外さないこと。** blind の質問から
「どちらが target に似ているか」を外した理由は「客観で測ってある（0.4981 対 0.5912）」
でしたが、**その 0.4981 が上限混入による誤り**でした（訂正後 0.5899 でほぼ同等）。
**壊れた測定は、その値だけでなく、それを根拠に外した観測も道連れにします。**
外すなら、根拠にした測定が**本物の入力を一度通してある**ことを確かめてから。
- **2 つの測定が食い違ったまま並んでいたら、どちらかが壊れています。** 同じ量を別経路で
測った値が 0.5676 と 0.4910 で食い違っていたのに、両方を文書に載せたまま進めました。
- **M5 の測定はすべて NHVSing V3 で行いました。** 2026-09-13 に上流の
**V3.1**（倍音のにじみと高域の縞を修正）を取り込み、config とツールの既定を
`nhv_v3_1.onnx` へ切り替えましたが、**`out/m5/` の数値は V3 のものです**。
上限（`*_vocoder_only.wav`）がボコーダーごと変わるので、**V3.1 で測り直した値と
混ぜないこと**。混ぜるなら両系を同じボコーダーで測り直します。
- **CSV の列名に説明を書くと、その列が引けなくなります。** `sheet.csv` のヘッダは
`pair,clip,vote  # vote に A / B / tie を書く` で、`csv.DictReader` は 3 列目の名前を
**説明ごと**受け取ります。`row["vote"]` は `None` になり、**26 票すべてが未記入と読まれ、
例外も出ませんでした**（実際に踏みました）。`tools/blind_test.py` の `read_sheet()` は
`vote` で**始まる**列を採り、無ければ落とします。**記入済みの入力を一度も通していない
読み取りコードは、通るまで壊れているかどうか分かりません。**
- **blind test の参照に上限を入れないこと。** 「正解の音」として
`*_vocoder_only.wav` を聴かせたくなりますが、これは **LeapSVC 自身のボコーダー
（NHVSing）の出力**なので、耳がその癖を覚えると **A / B のどちらが LeapSVC かを当てられます**。
参照に使えるのは**どちらの系の出力でもないもの**だけです（target 本人の録音、両系で同一の
変換元）。`tools/blind_test.py` の `listen_page()` は上限のパスと blind ディレクトリ外の
パスを拒否します（後者は**パスに system 名が出る**ため）。
- **持ち込み音源は学習分布より大きいので `--match-loudness` を付けること。** 配信用に整えられた音源は peak 1.0 付近まで上げられており、学習素材（波音リツ DB は peak 0.107）から大きく外れます。実測で loudness 条件が **+1.40σ** に出て、spectral centroid が上限比 **−47%** まで落ちました。合わせると **−24%** で他の素材と同じ範囲に戻ります（`preprocess/svc/loudness.py` の `loudness_match_gain`）。
- **推論時に入力の音量を勝手に触らないこと。** 学習（`preprocess.svc.run`）は生の音量のまま特徴を取ります。推論側で peak 正規化すると loudness 条件が学習分布からずれ、モデルが低域を持ち上げて高域を削ります（波音リツ DB は peak 0.107 なので実質 19 dB の増幅になり、spectral centroid が 620 → 368 Hz に落ちました）。`features_to_item()` は正規化の同一性を保証しますが、**その手前で波形を加工すると保証の外**です。
- **内容指標だけで音の劣化を判断しないこと。** 上の不具合で centroid が 620 → 368 Hz に落ちても、content cos は 0.8217 → 0.8096 としか動きませんでした。`tools/audio_metrics.py` の帯域指標を併せて見ます。
- **phrase 名の衝突は例外になりません。** `preprocess.svc.run` は曲ごとの通し番号で採番し、衝突を検出したら止めます。この採番を「ファイルごとに 0 から」に戻すと、同じ曲名の別ファイルが cache を**黙って上書き**し、shard の phrase 数が減るだけになります（GTSinger で 1,922 ファイルが 3 名に潰れました）。
- **曲名を ASCII に削らないこと。** `_SAFE` は `[^\w-]` なので CJK を残します。ASCII だけにすると日本語題の曲がすべて同じ名前になり、曲単位 split が効きません（1,922 中 1,723 件が潰れました）。曲名は casefold して表記ゆれ（`Heartful_Song` と `Heartful_song`）も畳んでいます。
- **`uv sync --extra <名前>` は「これだけにする」指定です。** 足す指定ではありません。
`uv sync --extra eval` だけを走らせると **train / export / dev が消えます**（実際に踏み、
tensorboard・onnx・ruff が消えました）。**必要な extra を毎回すべて並べること。**
- **上限を共有できない指標は、系をまたいで比べられません。** CER・信号品質・明るさは
「GT mel を**自分の**ボコーダーに通した再合成」が基準なので、Seed-VC と並べられません
（実測で確認）。**比べられるのは source か target 録音を基準にする指標だけ**です
（話者類似度・timing・F0・V/UV）。failure taxonomy の本数を 2 系で比べるのも誤りです
（測った軸の数が違う）。
- **hold-out 区間は有声率を確かめてから使うこと。** 曲を等分して窓内でずらすと**イントロや
間奏を掴みます**（実測で 6 本中 3 本が有声率 3.5〜34%）。**リツ本人の録音がリツの参照と
0.39 しか一致しない**という異常で気づきました（曲の中央なら 0.77〜0.79）。
`tools/m5_testset.py` は有声率 60% 以上を要求します。
- **M5 の客観指標は 4 つとも「上限との差」で読みます。** `tools/svc_convert.py --self-check` が
出す `*_vocoder_only.wav`（GT mel をボコーダーに通した再合成）が上限で、`asr_cer.py` と
`signal_quality.py` は**上限が無ければ実行を拒否**します。絶対値を品質として報告しないこと。
- **ボコーダーの出力は run ごとに変わります（2026-09-15 実測）。** NHVSing の ONNX に
**`seed` 属性の無い `RandomNormalLike`** があり（node `node_randn_like`）、ONNX Runtime が
毎回別の乱数を引きます。**上限（`*_vocoder_only.wav`）も変換結果も bit 一致しません** ――
**CPU + `torch.use_deterministic_algorithms(True)` でも一致しません**（実測。device も
決定的モードも無関係）。実害として、同じ V3 の 2 つの run で **26 clip 中 6 本の上限 CER が
5 点を超えてずれました**（`holdout05` は 100% → 11.3%、`unseen08` は 266.7% → 3233.3%）。
**ASR は離散なので、わずかな音の差が書き起こしを丸ごと別物にします** ―― 連続量（SQUIM・
centroid）にこの感度はなく、**CER に強く出ます**。
**対処は `tools/svc_convert.py --ceiling-from <dir>` で上限を使い回すことだけです**
（seed を渡す口がありません）。`convert.json` の **`ceiling_comparable` が `false` の記録
どうしは「上限との差」を比べられません**。
- **CER は言語ごとに割ってから読むこと（2026-09-15 実測）。** M5 の test set は
**日本語 6 / ラテン語 10 / イタリア語 5 / 英語 5** の混合で、pooled の差 **+16.8 点**は
言語別に **日本語 +4.8 / 英語 +13.1 / ラテン語（読めない）/ イタリア語（n=1）** でした。
**ラテン語は上限自身が 23.2% 揃いません**（source `ドナのビスパーチェ` 対 上限
`どなのびすパーチェ` = 同じ音の表記違い）。**上限が揃わない素材で差を出すと、表記の揺れを
内容の劣化として計上します。** `tools/cer_breakdown.py` が言語別に割り、上限中央が
10% を超える群と 3 本未満の群には差を出しません。
- **`ceiling_unusable`（上限 CER > 0.5）は clip ごとの安全弁で、素材ごとの安全弁ではありません。**
「上限が 23% 揺れる素材」は素通りします。**`--language` を素材に合わせるだけでは足りません** ―
実測では `ja` を渡しても**英語は英語で書き起こされ**（Whisper が音声から言語を判定する）、
崩れたのはラテン語とイタリア語でした。除外された 8 本の上限 CER は最大 **909%** です。
- **移調した男声 source は明瞭度も落とします（2026-09-15 実測）。** 英語 6 本のうち
男声 +12 半音の 2 本が **+16.0 と +100.0 点**（変換 CER 100%）、女声・移調なしの 4 本は
**+0.0〜+20.3 点**でした。**明るさと話者類似度だけの話ではありません。**
- **onset のずれは hop（5.8 ms）より細かく測れません。** 実測でずれの中央値がちょうど
1 フレームでした。**これは「ほぼずれていない」ではなく測定限界**です。timing で読むべきは
`matched_ratio`（実測 69.1%。onset の 3 割は対応が付かない）のほうです。
- **RTF は段ごとに出します。** 実測で **最大の項はボコーダー**です。CPU で acoustic 0.081 /
合計 0.654（ボコーダー 0.355）、**GPU では acoustic 0.006 / 合計 0.464（ボコーダー 0.432、
全体の 93%）**。NHVSing が ONNX で CPU 実行のためで、**acoustic を速くしても end-to-end は
ほとんど変わりません**（M6 の設計に直接効きます）。**README の性能表（SVS 経路・
Apple Silicon・ボコーダー < 0.1）とは機種も経路も違うので比較できません。**「1-step だから速い」は acoustic の
話であって pipeline 全体ではありません。`realtime_capable` は `rtf_total < 1` を見ているだけで、
chunk 境界も I/O 遅延も連続運転も見ていません。
- **信号品質（SQUIM）は話し声で学習されています。** 歌声への妥当性は未検証なので、
絶対値ではなく上限との差だけを読みます。
- SVC では online `pitch_aug` を使えません（特徴量が事前計算済みのため）。`train.py` が明示的に SystemExit します。augmentation は特徴量抽出前に行います。
- **学習はすべて vast.ai の Linux インスタンスで行います。手元の Windows で `train.py` を起動すると device によらず hook が止めます**（`tools/hooks/guard_commands.py` の `check_local_training`）。`--device cpu` に逃げるのも不可です。CPU は実測で 1 phrase 1200 step に約 60 分かかり、実験記録の環境も本番と食い違います。ローカル GPU は他の作業と取り合って `unspecified launch failure` を起こしました（実際に発生）。 手元の Windows 機は開発・推論・検証用で、セットアップは `tools/vast_bootstrap.sh`。API token 等は `.env`（gitignore 済み）に置きます。`uv.lock` は Linux も解決済みで、Linux では `triton` が入るため下の `torch.compile` の制約は当てはまりません。
- **Windows では `torch.compile` が使えません。** `harmonic_excitation.py` の倍音和は compile 前提の融合版（Linux + Triton で 3〜4 倍）ですが、Windows には Triton wheel がなく、さらに日本語ロケール（cp932）では inductor の template 読み込み自体が `UnicodeDecodeError` になります。`triton-windows` を入れても C コンパイラが必要です。コードは **compile 生成時と初回呼び出しの両方**でループ版へフォールバックします（数値差は加算順のみ）。最初から切るなら `LEAPSINGER_EXC_COMPILE=0`。
- **`eval_items` は話者ごとの本数です。** 3 話者なら 9 サンプルですが 23 話者では 69 になり、1 回の eval が mel 図と音声を 138 本書き出して 10 分以上（単一コア 100%）かかります。多話者では `eval_items: 1` にして `eval_interval` も大きくすること（実測で踏みました）。
- 1 曲しかない DB は eval split が空になります（`n_hold = min(eval_songs, 曲数 - 1)`）。`train.py` の `log_eval` は空なら評価を飛ばします。数フレーズの overfit 検証ではこの経路を通ります。
- RMVPE はマルチプロセスで動かさないこと（README の注意）。RMVPE の重み `preprocess/algorithms/rmvpe.pt` は初回実行時に HuggingFace から自動ダウンロードされます（約 181 MB、`.gitignore` 対象）。
- `dataset.py` の phrase 名は `{song}_{NNNN}` 形式で、`_song_of()` が曲単位の train/eval 分割に使います。この命名を崩すと leakage 防止が効かなくなります。
- ライセンス境界: コードは MIT ですが、同梱ボコーダー ONNX、Release 配布の学習済みモデル、学習に使った歌声 DB は MIT 対象外です。Seed-VC（GPL-3.0）は外部 baseline として実行するだけで、コードをこのリポジトリへ取り込まないこと。
