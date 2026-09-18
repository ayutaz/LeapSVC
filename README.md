# LeapSVC

[![CI](https://github.com/ayutaz/LeapSVC/actions/workflows/ci.yml/badge.svg)](https://github.com/ayutaz/LeapSVC/actions/workflows/ci.yml)

**English**: [README.en.md](README.en.md)

LeapSVC は**歌声変換（SVC）**の音響モデルです。歌唱の WAV を受け取り、**target 歌手の声で
歌い直した mel** を出します（mel から波形は NHVSing ボコーダー）。歌声合成モデル
[LeapSinger](https://github.com/wavtechyukky/LeapSinger) の励起（harmonic + noise）と
rectified flow をそのまま使い、**条件だけを「音素 + 持続長」から「content feature + F0」へ
差し替えた**ものです。

    SVS: 音素 + 持続長 + F0                      -> LeapSinger -> mel + F0 -> NHVSing -> WAV
    SVC: source WAV -> content / F0 / UV / 音量  -> LeapSVC    -> mel + F0 -> NHVSing -> WAV

> **このリポジトリは [LeapSinger](https://github.com/wavtechyukky/LeapSinger) から派生し、
> 独立して開発しています**（trunk は `main`）。
> **既存の SVS 経路は変更していません**（後半の「SVS 経路」節。上流と同じ内容です）。
> **学習済みの SVC 重みは配布していません** — 学習素材のライセンスが未解決のためです
> （[ライセンス](#ライセンス)）。

## LeapSVC とは

SVS と SVC の**分岐は condition encoder だけ**で、励起・flow backbone・損失・GAN・
checkpoint 形式・NHVSing 互換の mel はすべて共有します。

| | SVS（LeapSinger） | SVC（LeapSVC） |
|---|---|---|
| 入力 | 音素 + 持続長 + F0 | content + F0 + V/UV + 音量 |
| encoder | `PhonemeEncoder` + `LengthRegulator` | `ContentAdapter` |
| モデル | `HarmonicAcousticModel(MultiSpk)` | `HarmonicSVCModel`（`model.arch: svc`） |
| 特徴量 | `shard.npz`（音素ラベルが必要） | `svc_shard.npz`（**WAV だけで作れる**） |

- **content encoder は ContentVec** — 768 次元（layer 12）から固定ランダムに選んだ **256 次元**を
  shard に書きます。生の 768 は 1 段目の cache に残るので、部分集合を変える ablation は
  2 段目の再実行だけで回せます。
- **F0 は RMVPE。** SSL（50 Hz）から mel grid（172.265625 Hz）への整列は **left（直前保持）**です
  （比が整数にならないので、方式そのものがデータ契約です）。
- **F0 を条件として直接与える設計**なので、F0 追従・V/UV・timing は素直に出ます（下の比較表）。
- 出発点 `x0` が F0 由来の擬似 mel なので、**flow は周期成分をゼロから描く必要がありません**。

## いまどこまで来ているか

**確認済み。** 23 話者・約 18 時間の multi-singer base を 60,000 step 学習し、そこから
波音リツへ fine-tune（GAN あり）しました。**日本語の素材は 7 名・約 21.9 時間**です。

**読み方:** 明瞭度・信号品質・明るさは**「上限との差」でしか読みません**。上限とは
**正解の mel を同じボコーダーに通し直した音**（`--self-check` が出す `*_vocoder_only.wav`）で、
ボコーダーで到達できる限界です。**絶対値を品質として報告しません。**

| 層（すべて日本語・同一条件） | 明瞭度（CER の上限との差・差の中央値） | 話者類似度の回復率 | 明るさ（上限比） |
|---|---:|---:|---:|
| 学習した曲そのもの（診断用） | +1.0 点 | ― | ― |
| 未知曲・自己再構成 | +10.7 点 | 91.4% | 1.02x |
| 未知曲・既知話者 | +8.9 点 | 90.8% | 0.95x |
| **未知話者**（base に入っていない 2 名、n=20） | **+7.3 点** | **90.4%** | **0.93x** |

**未知話者でも既知話者と同等以上**でした（2026-09-17、音量を合わせた条件）。

### Seed-VC との比較（26 clip・同一 test set）

| 指標 | LeapSVC | Seed-VC | 判定 |
|---|---:|---:|---|
| 話者類似度（ECAPA-TDNN、20 秒） | 0.5899 | 0.5912 | ほぼ同等（相対差 0.2%） |
| F0 相関 | 0.9994 | ― | LeapSVC が上 |
| V/UV 一致 | 99.0% | ― | LeapSVC が上 |
| timing（onset の一致率） | 73.7% | 70.8% | LeapSVC が上 |
| blind preference（N=1 の非公式、26 ペア） | 4 | **21** | **Seed-VC が上**（引き分け 1） |

**「Seed-VC より良い」とは書けません。** blind preference で負けているためで、これは測る前に
登録した判定規則です。**差は未知 source に集中**し（20 ペアで 2 対 18）、**target 本人の
hold-out では拮抗**します（6 ペアで 2 対 3・引き分け 1）。

**負けた理由は測れています。** Seed-VC は 26 clip すべてで LeapSVC より明るく（spectral
centroid 比の中央値 1.80）、明るい音は同じ RMS でも大きく・近く聴こえます。ただし
**target 本人の録音は 1160 Hz、LeapSVC は 1067 Hz でほぼ一致**し、Seed-VC は 2149 Hz です。
**明るくすれば選好は取れますが target からは離れる**ので、モデル側では足していません。

**系をまたいで比べられるのは、source か target 録音を基準にする指標だけ**です（話者類似度・
timing・F0・V/UV）。**CER・信号品質・明るさは比べられません** — 上限が「自分のボコーダー」に
依存するためです。

### 測れていないこと

- **音質そのもの。** 内容保持・F0 追従・V/UV・明るさ・信号品質は測っていますが、音質では
  ありません。信号品質（SQUIM）は**話し声で学習されたモデル**なので、歌声での妥当性は未検証です。
- **男声・低音の未知話者。** 評価に使った未知話者 2 名は**女声で target とほぼ同音域**で、
  **C3 より下の滞在時間はどの素材にもほぼありません**。
- **リアルタイム。** `realtime_capable` は `rtf_total < 1` を見ているだけで、chunk 境界・
  audio I/O・連続運転を測っていません。**「リアルタイム」とは書きません。**
- **話者類似度の主観評価。** 客観では同等（相対差 0.2%）と分かっており、聴取では判定に
  至りませんでした（判定 4 票すべてが「後に聴いた側」）。

### RTF（実測・20 秒のフレーズ）

| 段 | CPU | GPU |
|---|---:|---:|
| acoustic（flow 16 step） | 0.081 | 0.006 |
| **ボコーダー（NHVSing / ONNX・CPU 実行）** | 0.355 | **0.432** |
| 合計 | 0.654 | 0.464 |

**最大の項はボコーダーで、GPU では全体の 93% です。** acoustic を速くしても end-to-end は
ほとんど変わりません。**「1-step だから速い」は acoustic の話**であって pipeline 全体では
ありません（**SVC 経路の既定は 16 step です**。下の表）。

## 使い方（SVC）

### 環境構築

Python は **3.13 固定**で、依存の管理と実行は [uv](https://docs.astral.sh/uv/) に統一しています。

    git clone https://github.com/ayutaz/LeapSVC
    cd LeapSVC
    uv sync --extra train --extra export --extra dev              # 開発・学習・書き出し
    uv sync --extra train --extra export --extra dev --extra eval # 評価も回すとき

**`uv sync --extra <名前>` は「これだけにする」指定です**（足す指定ではありません）。必要な
extra は毎回すべて並べてください。実行は `uv run python ...`、依存の追加は `uv add <package>`
です（素の `python` / `pip`、`uv pip` は使いません）。PyTorch・RMVPE・ボコーダーについては
後半の「SVS 経路」の環境構築を参照してください（共通です）。

### 前処理（WAV から特徴量へ）

**音素ラベルは要りません。** WAV のディレクトリから直接 shard を作ります。
**話者ごとにディレクトリを分けてください**（speaker id をディレクトリ名から引くため、
1 つの shard に複数話者を混ぜると区別できません）。

    uv run python -m preprocess.svc.run --wav-dir download/ritsu --out data/ritsu_svc

    # 入れ子の深いコーパス（<技法>/<曲>/<Group>/NNNN.wav）には --song-parts が必要
    uv run python -m preprocess.svc.run --wav-dir download/gtsinger/Japanese/JA-Soprano-1 \
      --out data/JA_Soprano_1 --song-parts 1 --max-hours 0.75

**2 段構成です。** 1 段目（重い・GPU）が ContentVec と RMVPE を回して `_cache/` へ、2 段目
（軽い・CPU）が整列・正規化・次元削減をして shard を書きます。`--from-cache` で 2 段目だけを
回せるので、**補間方法や 256 次元の選び方を変える ablation に重いモデルの再実行が要りません**。
**再実行で bit 一致します。**

**`--song-parts` を忘れないこと。** 既定は親ディレクトリ名を曲名にするので、深い配置では
曲名が潰れます。曲名は曲単位の train/eval 分割に使うため、潰れると leakage します。

### 学習

    uv run python -m train --config configs/svc_base.yaml \
      --data_dirs data/<話者>... --run_name svc_base_01 --out_root log --device cuda

    # target への fine-tune（GAN あり）
    uv run python -m train --config configs/svc_target_ft_gan.yaml \
      --data_dirs data/ritsu --init_from ckpt_060000.pt --finetune \
      --run_name svc_ritsu_ft_gan_01 --out_root log --device cuda

**同じ `--run_name` を使うと最新の checkpoint から黙って自動再開します。** 別の実験は必ず
名前を変え、base checkpoint を上書きしないでください。SVC では online の `pitch_aug` が
使えません（特徴量が事前計算済みのため。`train.py` が明示的に止めます）。

### 変換（推論）

    uv run python tools/svc_convert.py --wav <source.wav> --out out/<name> \
      --ckpt log/<run>/ckpt_015000.pt --manifest data/ritsu/manifest.json \
      --spk-id 0 --num-steps 16 --self-check --match-loudness --device cpu

- `--manifest` は **その checkpoint を学習したときの** manifest を渡します。正規化統計が run
  ごとに違うので（実測で `loudness_mean` が −4.6289 / −4.6554 / −4.7417）、**揃えると別の
  実験になります**。
- `--self-check` が**上限**（`*_vocoder_only.wav`）を書き出します。指標は上限との差で読むので、
  **必ず付けてください**。
- **持ち込み音源には `--match-loudness`** を付けます（下の表）。
- **上限は作り直さず使い回します** — `--ceiling-from <既存の出力ディレクトリ>`。NHVSing の
  ONNX に seed の無い `RandomNormalLike` があるため、**同じ入力でも run ごとに出力が変わります**
  （CPU + 決定的モードでも一致しません）。`convert.json` の `ceiling_comparable` が
  `false` の記録どうしは「上限との差」を比べられません。

### 評価

    uv run python tools/asr_cer.py           --dir out/<name> --language ja --device cpu
    uv run python tools/cer_breakdown.py     --cer out/<name>/cer.json --testset out/m5/testset.json
    uv run python tools/speaker_similarity.py --converted out/<name> \
      --target download/ritsu --unrelated .m0data/unrelated_ref --seconds 20 --device cpu
    uv run python tools/timing_metrics.py    --dir out/<name>
    uv run python tools/signal_quality.py    --dir out/<name> --device cpu
    uv run python tools/rtf.py --wav <vocal.wav> --ckpt <ckpt> --manifest <manifest> --device cpu

**話者類似度は encoder とクリップ長の両方に依存します。** ECAPA-TDNN を **12 秒以上**で使って
はじめて、事前登録した合格条件を満たします（`tools/speaker_similarity.py` は 12 秒未満の
クリップを拒否します）。**encoder を替えたら `tools/speaker_calibrate.py` で較正をやり直して
ください。**

**CER は言語ごとに割ってから読みます。** `tools/cer_breakdown.py` は上限の中央値が 10% を
超える群（表記の揺れが内容の劣化として計上される素材）と、3 本未満の群には差を出しません。

### テスト

    uv run python tools/smoke/run_smoke.py     # 全経路の疎通（合成音声・GPU で約 3 分）
    uv run python -m unittest test_svc_model test_svc_preprocess test_svc_dataset test_svc_metrics
    uv run ruff check .

単体テストは **514 件**で、重いモデルもネットワークも使いません。`run_smoke.py` の入力は
合成波形なので、**品質の検証にはならず**、配線が壊れていないことだけを示します。

## 既定値と注意点（すべて実測）

| 事項 | 値 / 対処 | 根拠 |
|---|---|---|
| flow の step 数 | **16**（`tools/svc_defaults.py`） | 掃引で決定。明るさは 1 step と同等ですが、細部（比 0.755 → 0.957）と話者性（回復率 78.2% → 90.6%）は明確に上。費用はほぼ無視できます（flow の RTF 0.044 → 0.065） |
| 低い声からの移調 | **+7 半音**（`SVC_TRANSPOSE_LOW_VOICE`） | 掃引で決定。**それまでの +12 は両方の軸で劣ります**（回復率 90.8% → 81.6%、CER の上限との差 +12.5 → +27.0 点）。n=8・1 話者・1 target なので、別の声では測り直すこと |
| 持ち込み音源 | **`--match-loudness` を付ける** | 学習は生の音量で特徴を取ります。配信用の音源は peak 1.0 付近まで上げられており（波音リツ DB は peak 0.107）、明るさが上限比 −47% まで落ちます。**プロのスタジオ録音でも必要**でした（0.71x → 0.93x） |
| 入力の音量 | **推論側で peak 正規化しない** | 同じ理由です。正規化するとモデルが低域を持ち上げて高域を削ります（spectral centroid 620 → 368 Hz）。**内容指標では検知できません**（content cos は 0.8217 → 0.8096 しか動かない） |
| ボコーダーの版 | **SVC は V3.1 据え置き**（`checkpoints/nhv_v3_1.onnx`）。V3.2 も同梱しています | 上限はボコーダーごとに変わるので、V3.2 へ移すと**この README のすべての「上限との差」が測り直し**になります。V3.2 は**高音域でフレーム単位に波形が急激に弱まる現象**を直しており、**聴取で報告された「音量が揺れる」に対応する可能性があります** — 移行は測り直しとセットで行います |
| 上限 | **1 度作って `--ceiling-from` で使い回す** | ボコーダーの出力が run ごとに変わるためです。実測で 26 clip 中 6 本の上限 CER が 5 点を超えてずれました（ASR は離散なので、わずかな音の差で書き起こしが別物になります） |

**明瞭度の律速はデータの量と多様性です。** `num_steps`・GAN の有無・fine-tune の step 数・
base の学習量・**話者の既知性**のいずれも明瞭度を動かさず、**動くのは「その曲を学習で見たか」
だけ**でした（学習曲 +1.0 点 対 未知曲 +7.3〜+11.1 点）。**`eval/loss` は明瞭度の代理に
なりません**（継続学習で 0.02311 → 0.01459 と下がっても明瞭度はほぼ不変）。

**過平滑には GAN が効きます。** `configs/svc_target_ft_gan.yaml` で fine-tune すると未知
source の話者類似度の回復率が 69.4% → 75.3% へ改善し、**trade-off は出ません**。ただし
**明瞭度は動きません**（非 GAN の fine-tune と +13.2 対 +12.7 点でほぼ同じ）。

要件、設計、データ/GPU、学習、評価、先行研究・ライセンス、実装状況、出典を分割した
調査ドキュメントは [doc/svc.md](doc/svc.md) を索引として参照してください。

## 今後の課題

- **男声・低音の未知話者での検証** — いま測れているのは**女声・target とほぼ同音域**の 2 名だけ
  です。**C3 より下の滞在時間**がどの素材にもほぼ無いという欠落も解消していません。
- **データの量と多様性** — 明瞭度の律速はここでした（上記）。**この規模では +12.7 点が水準
  という可能性も残ります**。
- **リアルタイム student は未着手**です（2026-09-14 に直列経路から外し、起動条件つきに
  しました）。**ボコーダーが RTF の 93% を占める**ので、acoustic だけ速くしても end-to-end は
  変わりません。
- **多言語対応** — 現状は日本語データで検証していますが、設計自体は言語に依存しません。
  他言語の音素辞書・データへの対応を行い、1つのモデルで複数の言語を扱えることを目標とします。
- **さらなる品質向上** — 擬似melの生成方法やパラメータの調整などで話者再現性に改善の余地が
  あると考えています。

## SVS 経路（音素 + 持続長 → mel）

以下は fork 元の [LeapSinger](https://github.com/wavtechyukky/LeapSinger) の内容です。
**この経路には手を入れていません。** デモも上流のものです。

LeapSinger は、CPUでも極めて高速かつ、安定した周期性成分の生成を行える歌声合成のためのDiffusion系の音響モデルです。音素・その長さ（持続長）・音高（F0）を受け取り、メルスペクトログラムを生成します。学習には音素タイミングを記した音素ラベルと音声のみを必要とし、言語による依存はありません（ただし、サンプルには全て日本語のデータを用いています）。

![LeapSinger の概要](doc/fig/leapsinger_overview.png)

**▶ デモを聴く： https://wavtechyukky.github.io/LeapSinger/demo/**

### LeapSinger とは

多くの拡散モデルは、ランダムなノイズから出発して、何ステップもかけて少しずつmelを描いていきますが、LeapSinger は**F0 から作った「擬似mel」（インパルス波形にホワイトノイズを足したもの）を出発点にして、rectified flow で1ステップでmelを仕上げます。** （綺麗な周期的成分の生成方法の学習に時間をかけることなく、一足飛びに高品質なmelを生成するためLeapSingerと命名）

擬似melはv/uv対応モデルと非対応モデルで異なっており、v/uv対応モデルは意図的に無声区間を作ることができます。

![v/uv あり・なしの擬似mel](doc/fig/pseudo_mel_vuv.png)

非対応モデル（上）は全フレームに倍音を敷きますが、対応モデル（下）は無声フレームで倍音をゲートします（暗い縦帯が無声区間）。

多くの検討の結果、高品質な歌声合成用のニューラルボコーダーはmelの質感に敏感であり、かつ、音響モデルは周期的成分を綺麗に生成することが課題であることが分かりました。LeapSingerは出発点がノイズではなく、すでに音高の形を持った擬似melなので、最も困難な周期的成分の描写の学習を避けることができます。この設計から、次の効果が得られます。

- **高品質な周期的成分** — ノイズの少ない高品質な周期的成分を安定して描画することができます。これは、ニューラルボコーダーによる合成結果の質感や話者の再現性に良い結果をもたらします。
- **速さ** — Reverse stepは1回のみであり、1コアのCPUで生成を行ってもRTFは0.03を切ります。なお、ステップ数を増やすことはできますが、1step以上行うとかえってGround truthから離れていきます。（**これは既存の歌声合成（SVS）経路の話**です。**SVC 経路は掃引の結果 16 step を既定にしています** — 明るさは 1 step と同等ですが、細部と話者性は 16 step が明確に上でした。`tools/svc_defaults.py` を参照）

このほかの機能:

- **多話者対応** — 話者IDで声を切り替えられます。例として、日本語3話者（御丹宮くるみ/夏目悠李/波音リツ）のモデルを配布します。
- **スタイル変換** — 同じ話者で歌い方（スタイル）を切り替えられます。

### デモ

**https://wavtechyukky.github.io/LeapSinger/demo/**

- 3話者を学習したモデルで、GT（本物の録音）と生成結果を並べて比較しています。なお、単一の話者で学習させるとわずかにmelの話者の再現性が上がりますが、合成品質に大きな差は見られませんでした。
- スタイル変換のデモを載せています。

### 性能

CPUで計測したRTF（Real-Time Factor。小さいほど速く、1未満なら実時間より速い）です。

音響モデル1本あたりのRTFを、Python nativeとONNXでコア数ごとに比較したものです。

| コア数 | Python native | ONNX |
|:--:|--:|--:|
| 1 | 0.027 | 0.090 |
| 2 | 0.026 | 0.063 |
| 4 | 0.027 | 0.058 |
| 8 | 0.026 | 0.054 |
| 10 | 0.024 | 0.065 |

- **Python native** は1ステップなのでコア数にほぼ依存せず、1コアでも RTF 0.027（実時間の約37倍速）です。
- **ONNX** は onnxruntime のオーバヘッドで native より数倍遅くなりますが、それでも実時間の10倍以上の速さです。コア数は4〜8が最速で、全コア（10）ではかえって遅くなります。
- **NHVSing ボコーダー** は CPU で RTF 0.1 未満です（詳細は NHVSing のリポジトリを参照）。

（計測条件：Apple Silicon 10コア・onnxruntime CPU・約7秒のフレーズ・中央値。機種によって変わります。）

**この表は SVS 音響モデル単体の値です。** **SVC 経路の end-to-end は別物**で（上の RTF の節）、実測では **GPU で合計 RTF 0.464、うちボコーダーが 0.432（93%）**、acoustic は 0.006 でした（CPU では合計 0.654）。**acoustic を速くしても end-to-end はほとんど動きません。** また **`realtime_capable` が True でも「リアルタイム」とは書きません** — chunk 境界・audio I/O・連続運転を測っていないためです。

フレーム設定は44.1kHz・hop size256です。hop size512 には、隣り合う2フレームの平均を取ることで対応します。

### 構成

上の図の流れは、次のとおりです。

1. **入力** — 音素・持続長・F0（必要なら話者ID）。
2. **Encoder** — 音素を埋め込み、持続長にしたがってフレーム長へ引き伸ばし、F0 と話者を足してconditionを作ります。
3. **励起（Harmonic + Noise Excitation）** — F0から、インパルス波形にホワイトノイズを足した「擬似mel」を作ります。これがflowの出発点です。
4. **Rectified Flow（1ステップ）** — 擬似メルを出発点に、conditionを条件にして、1ステップで本物らしいmelへ変換します。
5. **NHVSing** — melを音声（波形）に変換する、CPUでRTF0.1を切り、高音質なニューラルボコーダーです。 https://github.com/wavtechyukky/NHVSing/

擬似melは、倍音の減衰・本数・ホワイトノイズの強さを調整できます。しかしながら、検討を重ねた結果、インパルス波形はナイキスト周波数まで重ね切った方が品質に貢献します。

### 使い方

#### 環境構築

Python 3.13 固定です（`pyproject.toml` の `requires-python` と `.python-version` の両方で 3.13 に固定）。依存の管理と実行は [uv](https://docs.astral.sh/uv/) に統一しています。

    git clone https://github.com/ayutaz/LeapSVC
    cd LeapSVC
    uv sync                               # 推論・再合成・ノートブック
    uv sync --extra train                 # 学習（TensorBoard を追加）
    uv sync --extra export                # ONNX 書き出し
    uv sync --extra train --extra export  # 全部入り

Python の実行は `uv run python ...`、依存の追加は `uv add <package>` です（素の `python` / `pip`、`uv pip` は使いません）。`uv sync` は `.python-version` の Python を自動で用意し、`uv.lock` の解決結果どおりに `.venv/` を作ります。

- **PyTorch** は `pyproject.toml` の `[[tool.uv.index]]` で CUDA 版 wheel（cu130）を指定しているので、Windows / Linux では `uv sync` だけで GPU 版が入ります（PyPI の Windows 版 torch は CPU ビルドのため、この指定が必要です）。別の CUDA を使うときは url の `cu130` を `cu126` / `cu128` / `cu132` などに差し替えて `uv lock` をやり直してください。macOS は marker で PyPI の CPU/MPS ビルドに落ちます。
- GPU が見えているかは次で確認できます。

      uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
- **F0抽出（RMVPE）** の重みは初回実行時に自動ダウンロードされます（HuggingFace → `preprocess/algorithms/rmvpe.pt`）。
- **ボコーダー（NHVSing）** は `checkpoints/` に ONNX 同梱済みで、追加ダウンロード不要です。
- 学習・配布用の**音響モデル本体は Release で配布**しています（リポには含みません）。

#### config設定

学習や書き出しの設定は yaml で行います（`configs/` に例があります）。yaml は `mel` / `model` / `excitation` / `train` / `gan` / `data` の6つのセクションに分かれています。主な項目は次のとおりです。

- `model` — `spk_dim`（0より大きいと多話者）、`n_speakers`、`n_styles`（0より大きいとスタイル機能を使う）、`use_uv`（v/uvを条件に使うか）
- `excitation` — `n_harm`（倍音の本数）、`harm_decay`（倍音の減衰）、`noise_ratio`（ホワイトノイズの強さ）
- `train` — `lr`、`max_updates`、`num_steps`（推論のステップ数。**1** を推奨）、`balance_speakers`（話者を均等にサンプリング）など
- `gan` — 質感を鮮明化する GAN の設定。`enabled`（`false` で flow損失＋mel損失のみの学習）、`gan_start_step`（GAN を入れ始める step）、`gan_strength`（敵対的損失の強さ）など
- `data` — `spk_map` / `style_map`（データセットのフォルダ名 → 話者ID / スタイルID の対応）

#### 辞書の作成

日本語の音素の一覧は `dict/ja.phonemes` にあります（1行に1音素、並び順がそのままID、先頭の `pau` が ID 0、`#` から先はコメント）。日本語以外や独自の音素を使いたいときは、同じ形式のファイルを用意し、各コマンドに `--phonemes <ファイル>` で渡します（省略時は日本語の `dict/ja.phonemes` を使います）。

#### preprocess

データセット1つにつき、yaml（`configs/recipes/<db>.yaml`）を用意します。1曲あたり、音声 `wav` と音素タイミングの `.lab`（＋譜面）を使います。次のコマンドで `data/<db>/` に前処理済みのデータができます。

    uv run python -m preprocess.run --recipe configs/recipes/<db>.yaml

例に使った3つのデータベースは、次のスクリプトでダウンロードできます（各DBの規約に従ってご利用ください）。

    uv run python preprocess/download_scripts/download_oniku.py
    uv run python preprocess/download_scripts/download_natsume.py
    uv run python preprocess/download_scripts/download_ritsu.py

F0の抽出にはRMVPEを使います（RMVPEはマルチプロセスで動かさないようご注意ください）。

#### training

学習は2段構成です。前半は flow損失＋mel損失で土台の声を学習し、後半で GAN を入れて質感を鮮明化します（切り替えは config の `gan` セクションで設定。`gan.enabled: false` なら flow損失＋mel損失のみ）。

    uv run python -m train --config configs/<name>.yaml \
      --data_dirs data/<db> [data/<db2> ...] \
      --run_name <name> --out_root log --device cuda

同じコマンドをもう一度実行すると、途中から自動で再開します。

#### export

    uv run python -m export.cli \
      --ckpt log/<run>/ckpt_050000.pt \
      --out export/<name> --model-name <name> \
      --variant diffsinger --hop 256 --speaker bake --spk-id 0

話者の指定（bake / embed / なし）など詳しい説明は、下の「ONNX への書き出し」を参照してください。

ノートブックで、書き出しから利用まで一通り試すことができます。必要なモデルは Release からダウンロードして `notebooks/sample_data/` に置いてください（詳しくは同フォルダの `place_model_here.txt`）。

    notebooks/export_and_use_onnx.ipynb

ノートブックでは、音響モデルを単一のONNXに書き出し、一通り動かします（音素＋持続長＋F0 → mel → 音声）。

#### ONNX への書き出し

`export/` は、チェックポイントを自己完結した ONNXグラフに変換します。励起と1ステップのflowはグラフに焼き込まれるので、使う側は音素・持続長・F0を渡すだけで済みます。話者の扱いは、次から選べます。

- **焼き込み（bake）** — 1つの声を固定します。グラフがいちばん単純になります（話者入力なし）。
- **埋め込み（embed）** — 話者ベクトルを入力にします。1つのグラフでどの声にも切り替えられます。
- **なし（none）** — 単一話者モデル向けです。話者の入力も焼き込みもしません（話者の概念がないグラフ）。多話者モデルではbakeかembedを使います。

#### ボコーダー

`checkpoints/` に NHVSing ボコーダーを2つ同梱しています。

- `nhv_v3_2.onnx` — hop size256のmelとF0を受け取ります。
- `nhv_v3_2x.onnx` — hop size512のmelとF0を受け取ります。

`nhv_v3_1.onnx` と `nhv_v3_1x.onnx`（V3.1）も置いてあります。**SVC 経路の測定がすべて
V3.1 の上限を基準にしている**ため、測り直しが済むまで消せません。

既定は **V3.2** です（高音域でフレーム単位に波形が急激に弱まる現象を、LTV フィルタの重ね合わせを Hann 窓化して解消した最新のウェイト。入出力の仕様は V3.1 と同一なので差し替えるだけで使えます。詳細は [NHVSing](https://github.com/wavtechyukky/NHVSing/) を参照）。

### オプション

- **辞書** — 任意の音素辞書を指定できます。日本語以外にも対応できる設計です（多言語対応そのものは今後の課題です）。
- **v/uv の扱い** — 有声・無声（v/uv）を条件に使うモードと、隙間を線形補間で埋めた連続F0だけを使うモードを選べます。
- **励起の調整** — 倍音の減衰・本数・ホワイトノイズの強さを変えられます。
- **学習レシピ** — 学習の条件は yaml で指定します（使用するデータセット、話者ID、話者ごとのスタイルやデータなど）。

## ライセンス

コードは MIT です（`LICENSE`）。ただし、同梱のボコーダー ONNX（`checkpoints/nhv_v3_2*.onnx`）、および Release で配布する学習済みモデルとその学習に使った歌声データベースは MIT の対象外で、それぞれのライセンス・規約に従います（下の謝辞、およびモデル配布物の `CREDITS.txt` を参照）。

**SVC 経路の制約はさらに強くなります。** base モデルの学習に **GTSinger（CC BY-NC-SA 4.0、非商用かつ継承）** を使っており、ShareAlike が学習済み重みに及ぶかはライセンス条文からは決まりません。**そのため SVC の重みは配布していません**（研究・個人利用のみという決定。`doc/svc-dataset-ledger.md`）。また GTSinger は「本人の同意なく特定個人の歌声を生成すること」を禁じており、**声を変換するには target 歌手の同意が要ります**。ソフトウェアのライセンスとは別の話です。詳細は `LICENSE` の SVC 向け NOTICE を参照してください。

## 謝辞

本モデルの学習に使わせていただいたデータセットと、関連するプロジェクトに感謝します。

- 御丹宮くるみ歌声データベース（御丹宮くるみ） — https://onikuru.info/db-download/
- 夏目悠李（歌声DB制作: アマノケイ／音声提供者: 霧野蒼太） — https://ksdcm1ng.wixsite.com/njksofficial/enunu-nnsvs
- 波音リツ — https://www.canon-voice.com/voicebanks/
- Neural Homomorphic Vocoder — https://www.isca-archive.org/interspeech_2020/liu20_interspeech.html
- dsp（zjlww） — https://github.com/zjlww/dsp

SVC 経路では次も使わせていただいています。

- GTSinger（CC BY-NC-SA 4.0） — https://github.com/AaronZ345/GTSinger
- VocalSet（CC BY 4.0。未知 source の評価用） — https://zenodo.org/records/1492453
- ContentVec（MIT） — https://huggingface.co/lengyue233/content-vec-best
- RMVPE — https://arxiv.org/abs/2306.15412 （重みは lj1995/VoiceConversionWebUI から取得。**ライセンス未確認**）
- 東北きりたん（©SSS。**未知話者の評価用**。学習には使っていません） — https://zunko.jp/kiridev/login.php
- No.7（「小岩井ことり歌唱データベース」由来。楽曲は小岩井ことり氏に帰属／商用は No.7 応援委員会。**未知話者の評価用**。学習には使っていません） — https://voiceseven.com/

配布する多話者モデルには、各データベースの規約に従って上記のクレジットを表示します。夏目悠李については **歌声DB制作: アマノケイ／音声提供者: 霧野蒼太** を表示し、「夏目悠李の出力音声に関する利用規約」をモデル配布物に同梱します。
