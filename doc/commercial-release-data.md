# 商用利用可能なモデルとして配布するためのデータ要件

調査日: 2026-09-19

対象: `main`（SVS 経路と SVC 経路の両方）

> **この文書は法的助言ではありません。** 各行は調査日に取得した公開ページ・論文・リポジトリの
> 記載を要約したものです。実際に配布・販売する前に原文を読み、判断が分かれる点は権利者と
> 法務へ確認してください。問い合わせた場合は、その回答と日付を
> [データセット台帳](svc-dataset-ledger.md) へ追記します。

関連: 権利条件の一次資料は [データセット台帳](svc-dataset-ledger.md)、ライセンスの棚卸しは
[先行研究・ライセンス・リスク](svc-prior-art-license.md) 3 節、量と計算資源の実測は
[データと計算資源](svc-data-compute.md)。

## 1. 結論（先に読む）

**確認済み: 塞いでいるのはデータの量ではなく、2 つの部品の権利です。** 素材を足す前に、
この 2 つを作り直す必要があります。

| 塞いでいるもの | 理由 | 作り直しに要るもの |
|---|---|---|
| 同梱ボコーダー NHVSing の**重み** | 上流 README が**非商用と明記**（2026-09-19 再確認） | 権利クリーンな素材での再学習、または MIT の代替への差し替え |
| multi-singer base の**重み** | 素材の GTSinger が **CC BY-NC-SA 4.0**（非商用・継承） | 権利クリーンな素材での base 再学習 |

**確認済み: 逆に、いちばん高価そうな 2 つは既に解決しています。** content encoder
（ContentVec、MIT、LibriSpeech 960 h で学習）と F0 抽出（RMVPE、取得元の HF リポジトリが
MIT を宣言）は、**新たにデータを集めずに商用で使える見込み**です。**したがって新規に必要な
データは「44.1 kHz 以上の歌唱」に限られます**（3 節）。

**訂正（2026-09-21、実測）: 素材は「量としては」足りますが、それだけでは現行品質を
再現できませんでした。** 権利クリーンな 16 話者 / 17.0 h で base を学習し、現 base
（23 話者 / 18 h）と同条件で比べたところ、**明瞭度 +25.5 → +50.5 点、話者類似度の回復率
77.9% → 49.2%、明るさも悪化**しました（[実行計画](svc-plan.md) 14 節 S1）。
**律速は量ではなく「歌詞のある楽曲が無いこと」**という読みです ―― 商用可の 6.1 h を占める
VocalSet は**音階・技法練習で歌詞を含みません**。
**この読みは統制実験で支持されました**（[実行計画](svc-plan.md) 15 節 S1b。中身だけを歌詞つき実楽曲へ
替えると明瞭度の損失の 65% が戻る）。**ただし話者類似度と明るさは戻らず**、スタジオ品質と
1 歌手あたりの量も要ると見ています。**買うべき素材は約 20 歌手 × 0.75 h の歌詞つき楽曲**です。以下の「素材は足ります」は**量の話**として
読んでください。

**確認済み: いまの規模を権利クリーンに作り直すぶんなら、素材は足ります（2026-09-20 に実測）。**
商用利用可の歌唱は **約 29.2 時間 / 40 話者**で、**由来に懸念のある SingVERSE を外しても
20.1 時間 / 23 話者**です（4.1 節。時間は 2026-09-19 の見積もり 39 h から**減り**、
話者数は 24 から**増えました**。訂正の内訳は同節）。現在の base は **23 話者・約 18 時間**
なので、**同等規模の再構築は素材の面では可能**です。

**確認済み: その先（品質を上げる）には足りません。** 明瞭度の律速は実測で**データの量と
多様性**でした（[実行計画](svc-plan.md) M5、[CLAUDE.md](../CLAUDE.md) の落とし穴）。
次の段は 100 時間 / 50 話者級ですが、**公開素材ではそこへ届きません**。埋める手段は
自前収録・有償ライセンス・speech での話者数稼ぎ・augmentation の 4 つです（6 節）。

**確認済み: 手元の 1,000 時間級素材は、商用トラックには効きません。**
`tts-dataset/japanese-singing-voice`（約 1,000 h・日本語・YouTube 由来）と
`-vocal-only` は、**dataset card 自身が CC BY-NC 4.0 かつ「音声の著作権を保有していない」**と
述べています（4.5 節）。**商用トラックの量は 4.1 節の 39 時間から変わりません。**
**したがって現実的な形は 2 系統です** —— **非商用版（量で押せる。1,000 時間級）**と
**商用版（39 時間。作り直し）**。両者は**上限（ボコーダー）が違えば数値も比べられません**。

**見積もり: 費用はほぼ全額がデータ取得費です。** GPU は実測で、18 時間の base の継続 run
（30,000 → 60,000 step）が **$2.148**（うち通信 $0.868。
[データと計算資源](svc-data-compute.md) 8 節）でした。100 時間規模でも **数十ドル**に収まる
見込みで、**収録費・ライセンス料に対して誤差**です。

**要ユーザー判断: 権利の確定作業が先です。** 4 件の未解決（波音リツの機械学習条項、
RMVPE 重みの由来、SingVERSE の素材由来、CC BY-SA を base に入れるか）は、**素材を 1 時間も
集めなくても今日から進められます**（8 節の S0）。

## 2. 「商用利用可能」を 4 つに分解する

**決定（この文書での用語）:** ひとまとめにすると議論が混ざるので、4 層に分けます。
層ごとに必要な許諾が違います。

| 層 | 何が問われるか | 現在の状態 |
|---|---|---|
| L1 コード | 配布・改変・商用が許されるか | **MIT**（`LICENSE`）。問題なし |
| L2 実行時の依存物 | 同梱・実行する重みが商用可か | **NHVSing の重みが非商用**。ContentVec は MIT、RMVPE は取得元が MIT 宣言 |
| L3 学習済み重み | 学習素材の条件が重みに及ぶか | **GTSinger の NC・SA が上流にある**。SVS 側は日本語 3 DB の規約 |
| L4 生成音声 | 出力を商用に使えるか | 素材ごとに違う。波音リツは可、他は要許諾（[台帳](svc-dataset-ledger.md) 1 節） |

**確認済み: L3 と L4 は別です。** 「生成音声の商用利用が可」（波音リツ）でも、
その DB で学習した**重みを配ってよいか**は別の記載です。逆に重みを配れても、
出力の用途が縛られることがあります。**両方を素材ごとに確認する必要があります。**

**要ユーザー判断: どこまでを商用にするのかで必要な素材が変わります。**

| 目指す形 | L3 重み配布 | L4 生成音声 | 素材の条件 |
|---|---|---|---|
| a. SaaS として推論だけ提供 | 不要（社外に出さない） | **必要** | 重み配布可の確認は不要だが、**非商用素材は使えない**（商用サービスでの利用自体が商用） |
| b. 重みを無償公開（商用利用可） | **必要** | **必要** | 最も厳しい。CC BY / Apache / 個別許諾のみ |
| c. 重みを有償販売 | **必要** | **必要** | b と同じ + 再配布条件（SA の扱い） |

**注意: a でも非商用素材は使えません。** 「配布しないから大丈夫」は、**非商用（NC）条項には
効きません** —— NC が禁じるのは商業的利益を主目的とする利用そのものです。

## 3. データが要るのはどの部品か

**確認済み:** pipeline を部品に割ると、**新しいデータが要るのは 3 つだけ**です。

| 部品 | いま使っているもの | ライセンス | 商用で使えるか | 新しいデータが要るか |
|---|---|---|:-:|---|
| content encoder | ContentVec（`lengyue233/content-vec-best`） | **MIT**。LibriSpeech 960 h（CC BY 4.0）で学習 | **可** | **不要** |
| F0 抽出 | RMVPE（`lj1995/VoiceConversionWebUI` の `rmvpe.pt`） | 取得元が **MIT を宣言**。参照実装は Apache-2.0 | **要確認**（下記） | 不要（差し替えなら要る） |
| 音響モデル（SVC base） | GTSinger + 日本語 3 DB で学習 | **CC BY-NC-SA 4.0** が混在 | **不可** | **必要** |
| 音響モデル（target） | 波音リツ 3 音源 | サイト規約で商用可・再配布可 | **可**（機械学習条項は記載なし） | 必要（別 target なら） |
| ボコーダー | NHVSing V3.1 / V3.2 の ONNX | **重みは非商用** | **不可** | **必要** |
| 評価（話者類似度） | ECAPA-TDNN（speechbrain） | 配布物に**同梱しない**ので L2 に入らない | 影響なし | 不要 |

**確認済み: RMVPE の重みは「取得元が MIT を宣言している」までです。** 配布元
（`lj1995/VoiceConversionWebUI`）のモデルカードは `mit` を宣言し、参照実装
（`Dream-High/RMVPE`）は Apache-2.0 です。**ただしどちらも重みの学習素材を書いていません。**
宣言は上流の適法性を保証しないので、**商用で同梱する前に由来の確認が要ります**（9 節）。

**確認済み: mel 仕様がボコーダー差し替えの制約になります。** `mel` セクション
（44.1 kHz / hop 256 / 128 mel / 40–16,000 Hz）は前処理・loader・励起で共有される契約です
（[CLAUDE.md](../CLAUDE.md)）。差し替え候補は 2 つあり、**保てる作業量が違います**。

| 案 | 内容 | mel 契約 | 音響モデル | 再測定 |
|---|---|---|---|---|
| **V-a（推奨）** | NHVSing の**コード（MIT）**を権利クリーンな素材で再学習 | **維持** | **再学習不要** | 上限が変わるので**必要** |
| V-b | `nvidia/bigvgan_v2_44khz_128band_256x`（**MIT**、44 kHz / 128 mel / hop 256）へ差し替え | **変更**（fmax 22,050 Hz） | **全面再学習** | 必要 |

**推奨は V-a です。** BigVGAN v2 は sample rate・mel 数・hop が一致しますが **fmax が
22,050 Hz** で、40–16,000 Hz という契約と違います。mel 契約を変えると前処理・励起・
既存 shard・学習済み重みのすべてが作り直しになります。

**確認済み: どちらの案でも「上限との差」は全部再測定です。** 上限（`*_vocoder_only.wav`）は
ボコーダーの出力なので、ボコーダーを変えると基準が動きます。V3.1 → V3.2 の据え置きを
決めたときと同じ理由です（[CLAUDE.md](../CLAUDE.md) の落とし穴）。

## 4. 商用利用可の素材（2026-09-19 調査）

### 4.1 歌唱（44.1 kHz 以上）

**確認済み:** ライセンス欄は一次資料（公式ページ / Zenodo record / HF dataset card）の記載です。
時間と sample rate に「実測」と付けたものは、このリポジトリで取得して測った値
（[台帳](svc-dataset-ledger.md) 4b 節）。

| 素材 | 歌手 | 時間 | sample rate | 実効帯域 p50 | ライセンス | 楽曲か |
|---|---:|---:|---|---:|---|:-:|
| [VocalSet](https://zenodo.org/records/1442513) | **20** | **8.73 h** | **44,100** | 15,996 Hz | **CC BY 4.0** | ✗ 音階・技法練習 |
| [SingVERSE](https://huggingface.co/datasets/amphion/SingVERSE) | **17** | **9.07 h**（clean 側） | 44,100 | 未測定 | **CC BY 4.0**（**由来に具体的な懸念。下記**） | ○ |
| 波音リツ 3 音源 | 1 | **10.41 h** | **44,100** | 22,026 Hz | [サイト規約](https://www.canon-voice.com/terms/): 商用可・再配布可・クレジット不要 | ○ |
| [NIT-SONG070-F001](https://sinsy.sourceforge.net/readme_hts_voice_nitech_jp_song070_f001.php) | 1 | **0.526 h** / 31 本 | **48,000** | 17,364 Hz | **CC BY 3.0**（**DB 本体の `data/COPYING` で確認**） | ○ 童謡 |
| [PJS](https://sites.google.com/site/shinnosuketakamichi/research-topics/pjs_corpus) | 1 | **0.448 h** / 100 本 | **48,000** | 23,880 Hz | **CC BY-SA 4.0**。"Free for non-commercial and commercial use" | ○ 自作曲 |
| **合計** | **40** | **約 29.2 h** | | | | |
| **SingVERSE を除いた合計** | **23** | **約 20.1 h** | | | | |

**すべて実測です（2026-09-20）。** 時間・sample rate・実効帯域は取得して測った値で、
生データは `.m0data/p0/nit_audit.json` / `.m0data/p0/pjs_audit.json` /
`.m0data/p0/singverse_meta.json`（VocalSet と波音リツは[台帳](svc-dataset-ledger.md) 4b 節）。

#### 訂正（2026-09-19 の版から 3 件）

**黙って書き換えず、旧値と原因を並べます。**

| 項目 | 旧値（2026-09-19） | 新値（実測） | 原因 |
|---|---|---|---|
| SingVERSE の時間 | 18.14 h | **9.07 h** | **dataset card の 18.14 h は clean と noisy の両側の合計**でした。card の scenario 別表を合計すると 32,656 秒 = 9.07 h で、本数 3,929（pro 1,847 / non_pro 2,082）も parquet の実測と一致します |
| SingVERSE の歌手数 | 不明 | **17 名** | parquet の `singer` 列だけを読んで数えました（音声は落とさず、列射影のみ） |
| NIT-SONG070 の時間 | 約 1.2 h（ACE 論文 Table 1） | **0.526 h** | raw（48 kHz / int16 / LE）31 本のバイト数から算出。**論文表の値より半分以下**です |
| PJS の時間 | 1 h 未満と見込む | **0.448 h** | zip の `*_song.wav` 100 本を実測 |
| 合計 | 約 39 h / 24 話者以上 | **約 29.2 h / 40 話者** | 上記の差分。**話者数は増え、時間は減りました** |

**結論は生き残ります。** SingVERSE を外しても **20.1 h / 23 話者**で、現在の base
（23 話者・約 18 h）とほぼ同じです。**同等規模の再構築は素材の面では可能**という 1 節の判断は
変わりません。

#### SingVERSE の由来に具体的な懸念が出ました（2026-09-20）

**確認済み: `singer` 列に実在の商業アーティスト名が含まれます** ――
`TaylorSwift`（51 本）/ `zhoujielun`（周杰倫、46）/ `chenyixun`（陳奕迅、83）/
`dengziqi`（鄧紫棋、105）/ `zhangliangying`（張惠妹系、87）など。dataset card のファイル名例は
**`chenyixun-burubujian-Concert-pro-part_0.wav`**（陳奕迅「不如不見」）で、clean 側は
"studio-quality clean vocal reference" と説明されています。

**仮説: clean 参照は市販音源に由来する可能性があります。** そうであれば **CC BY 4.0 の宣言が
上流と整合しません**。**確認できるまで 4.1 節の合計に数えないほうが安全**です
（上の表で「SingVERSE を除いた合計」を併記したのはこのためです）。
**8 節 S0 の SingVERSE への問い合わせは、この 4 件の中で最も優先度が高い**と考えます。

**確認済み: いまの base（23 話者・約 18 時間）と同程度の規模が、権利クリーンな素材だけで
揃います。** ただし中身は違います。**日本語は波音リツ 1 名（10.41 h）と NIT-SONG070 1 名だけ**で、
現在の日本語 19.82 時間 / 5 名（[データと計算資源](svc-data-compute.md) 3 節）より薄くなります。

**確認済み: VocalSet を base に入れると、未知 source の test set が無くなります。**
いまは VocalSet を**学習に使わない**ことで汎化の過大評価を避けています（[台帳](svc-dataset-ledger.md)
2 節）。商用可の素材が少ないため、**学習と評価で同じ素材を使いたくなる圧力がかかります**。
**決定（2026-09-20。S1 の前に確定させました）:** VocalSet の 20 名を**性別で層別して**
**学習 14 / 評価 6** に分けます。**同じ歌手を両側に置きません**（曲単位 split では話者が漏ります）。

**新しいコードは書いていません。** 既存の `preprocess/svc/split.py` の `split_by_group()` が
**層別のラウンドロビン hold-out** に対応しているので、歌手を group、性別を strata として
`seed=42 / eval_groups=0 / test_groups=6` で呼ぶだけです（`random.Random` なので
バージョンを越えて再現します）。

| | 歌手 |
|---|---|
| **評価（未知 source）6 名** | `female1` / `female2` / `female6` / `male7` / `male9` / `male10`（**女 3 / 男 3**） |
| **学習（base）14 名** | `female3` `female4` `female5` `female7` `female8` `female9` / `male1` `male2` `male3` `male4` `male5` `male6` `male8` `male11`（女 6 / 男 8） |

**性別で層別する理由:** 男声が評価から消えると、**いちばん弱い条件（低音の source）を
測れなくなります**。実測で「必要な移調量は source の F0 に依る」と分かっているので、
評価側に男声を必ず残します。

**要確認（SingVERSE）:** 「実環境の録音」と「スタジオ品質の clean 参照」の対になった
enhancement 用ベンチマークです。**clean 側の素材がどこから来たか**を確認しないと、
**CC BY 4.0 の宣言が上流と整合しているか分かりません**。加えて**ベンチマークなので、
学習に入れると以後その指標で評価できません**。

### 4.2 speech（帯域と話者数を稼ぐ）

**確認済み:** 歌唱ではないので target 音色の素材にはなりませんが、**話者数・低音域・
ボコーダーの汎化**を稼ぐ候補です。歌唱素材は**全部が C3（約 131 Hz）より下をほとんど
持っていません**（実測で最大 3.6%。[台帳](svc-dataset-ledger.md) 4b 節）。
**男声 speech はそこを埋められます。**

| 素材 | 話者 | 時間 | sample rate | ライセンス |
|---|---:|---:|---|---|
| [VCTK 0.92](https://datashare.ed.ac.uk/handle/10283/3443) | **110** | 約 44 h | **48,000**（原録音 96 kHz） | **CC BY 4.0**（`license_text.txt` を取得して確認） |
| [AISHELL-3](https://huggingface.co/datasets/AISHELL/AISHELL-3) | **218** | 85 h | 要確認 | **Apache-2.0**（HF dataset card） |
| [Emilia-YODAS](https://huggingface.co/datasets/amphion/Emilia-Dataset) | 多数 | **113.9k h**（日本語 1.1k h） | 要確認（in-the-wild） | **CC BY 4.0**（Emilia 本体は CC BY-NC 4.0 なので**取り違えないこと**） |

**仮説: speech を混ぜると低音域の外挿が改善します。** 実測で「出力のスペクトル傾斜は入力の
F0 に強く従う」「低い声の source では移調が要る」ことが分かっており
（[CLAUDE.md](../CLAUDE.md)）、**必要な移調量を固定定数で賄えない**のは学習分布に低音が
無いためという読みです。**未検証です。**

**注意: sample rate を確かめてから足すこと。** 24 kHz 素材（Nyquist 12 kHz）は
12–16 kHz の mel bin が常に空になり、こもった出力を学習します。JVS-MuSiC を外したのと
同じ理由です（[台帳](svc-dataset-ledger.md) 3 節）。**Emilia-YODAS は YouTube 由来で
帯域が揃わない**ので、`preprocess/svc/audit.py` の `effective_bandwidth_hz()` で
実測して足切りします。

### 4.3 使えないと確定したもの（と理由）

**確認済み:** 規模の大きい歌唱コーパスは、ほぼ全部が非商用です。

| 素材 | 規模 | 不可の理由 |
|---|---|---|
| GTSinger | 80.59 h / 20 名 / 48 kHz | **CC BY-NC-SA 4.0**（現 base の中核） |
| ACE-Opencpop / ACE-KiSing | 128.9 h / 32.5 h | **CC-NC**。加えて合成音声 |
| OpenSinger | 50 h / 66–93 名 | **CC-NC**、かつ 24 kHz |
| M4Singer | 29.8 h | **CC-NC**、流通形態が 24 kHz |
| Children's Song Dataset | 100 曲 / 1 名 / 44.1 kHz | **CC BY-NC-SA 4.0**（NHVSing V3 の学習素材） |
| Opencpop | 5.2 h / 44.1 kHz | **CC-NC-ND**（ND = 派生禁止） |
| ccmusic-database/acapella | — | **CC BY-NC-ND 4.0**（NHVSing V3 の学習素材） |
| 東北きりたん / NUS-48E / NHSS / SingStyle111 | 1.0 h / 4.8 h / 4.8 h / 12.8 h | **研究用途限定** |
| [SingNet](https://arxiv.org/abs/2505.09325) | **3,000 h** | **音声は公開されていません**（sample pack と web 音源からの収集。公開物は学習済みモデルのみで、条件の記載なし） |
| [CP Singing Voice Dataset](https://zenodo.org/records/18773342) | — | **Ircam Forum License**（restricted。公衆への利用許諾なし） |
| JSUT-song | 27 曲 / 約 25 分 | 既定は**非商用**。商用は東大 TLO へ要相談。**再配布不可** |
| No.7（小岩井ことり歌唱DB） | デモ 50 曲 / 44.1 kHz 以上 | **商用は「ご相談ください」**。非商用は事前申請不要。**交渉の余地あり** |
| 御丹宮くるみ / 夏目悠李 | 1.42 h / 1.20 h | 生成音声の商用利用に**事前許可が必要**（[台帳](svc-dataset-ledger.md) 1 節） |

**確認済み: 「商用可の歌唱が少ない」のは 2026 年時点の市場全体の性質です。** ベンダーの
マーケティング資料（[The Vocal Market](https://thevocalmarket.com/blogs/enterprise/state-of-vocal-data-licensing-2026)、
**二次情報**）は「オープンソースの clean な歌唱は合計 **約 230 時間**で、大半が中国語。
英語の clean な歌唱は公開リポジトリには実質存在しない」と書いています。**独立に検証して
いませんが、4.1〜4.3 節の内訳と矛盾しません。**

### 4.4 取り違えの記録: Zenodo の CC BY は論文 PDF に付いていた

**確認済み（この調査で実際に踏みました）:** SingStyle111（44.1 kHz・スタジオ録音・
歌詞つき・12.8 h）の Zenodo record `10265401` は `license: cc-by-4.0` / `access_right: open`
です。**商用可の大型素材を見つけたと思いました。** API で中身を見ると、
**record に入っているのは `000091.pdf`（797 kB）1 本だけ**でした。CC BY 4.0 は
**ISMIR の論文 PDF** に付いており、音声には及びません。論文本文は
"freely available for research purposes" です。

**教訓: license は record ではなく「その license が掛かっているファイル」で確認すること。**
`https://zenodo.org/api/records/<id>` を引けばファイル一覧が出ます。**データセットの
landing page が論文 record を兼ねている例は珍しくありません。**

### 4.5 手元の 1,000 時間級素材（`tts-dataset/*`）: 非商用トラック限定

**確認済み（2026-09-19、HF の dataset card と `api/datasets` を取得）:** 利用者が自身で
アップロードした 2 つのデータセットです。

| 素材 | 規模 | 形式 | 由来 | 宣言ライセンス |
|---|---|---|---|---|
| [`tts-dataset/japanese-singing-voice`](https://huggingface.co/datasets/tts-dataset/japanese-singing-voice) | **約 1,000 h** / 15,311 ファイル / 73 GB / 78 shard | **MP3 約 170 kbps VBR**（WebDataset） | **YouTube**（公式 MV 約 40% / cover 約 30%）。`ayousanz/music-youtube-list` 由来 | **CC BY-NC 4.0** |
| [`tts-dataset/japanese-singing-voice-vocal-only`](https://huggingface.co/datasets/tts-dataset/japanese-singing-voice-vocal-only) | 約 199.6 GB / `vocals-0000..0039.tar` の 40 本（各 約 5.0 GB） | **44.1 kHz / stereo / PCM_16 の WAV**（実測。MP3 ではない）。1 ファイル = 1 曲まるごと | 上のボーカル分離。**同じ動画 ID と同じ sidecar JSON を持つ** | **CC BY-NC 4.0**、**gated**（手動承認） |

**確認済み: 商用トラック（この文書の主題）には使えません。** dataset card 自身が
"This dataset does not own the copyright to the audio files" と書き、**CC BY-NC 4.0** を
宣言しています。**アップロード主であることは、原盤（レコード製作者）・実演家・楽曲の権利を
持っていることを意味しません。** したがって 2 節の **L3（重み配布）と L4（生成音声の商用
利用）はこの宣言の外に出られず、4.1 節の 39 時間を置き換えられません。**

**要ユーザー判断（利用者の申告、2026-09-19）:** 「**自分がアップロードしたデータセットなので
使える。事前学習は法律的に問題ない**」との判断を受けています。**この文書はその判断を学習段階
（9 節の 30 条の 4 の枠組み）についてのものとして扱い、重みの商用配布については未解決のまま
にします。** 2 つは別の層です（2 節）。**判断が「商用配布も可」であれば、根拠（原盤・実演・
楽曲の許諾）をこの節に追記してください** —— そのとき 4.1 節の集計と 5 節の結論は大きく変わります。

**したがって位置づけは「非商用トラックの base を 1,000 時間級にする素材」です。**
現 base（約 18 h）の **55 倍**、GTSinger（80.59 h）の **12 倍**で、**しかも日本語**です。
[実行計画](svc-plan.md) M5 で「明瞭度の律速はデータの量と多様性」と結論した点に対して、
**手元にある唯一の量的な弾**です。

#### 期待できること / できないこと

| 見込み | 根拠 |
|---|---|
| 低音域の男声を埋められる → **棄却（2026-09-19、n=75）** | C3 未満の滞在率の**中央値は 1.224%** で、夏目悠李の 3.6% に届きません（[実行計画](svc-plan.md) 12 節 P0-2 の結果）。**事前登録した規則で不合格**。F0 p50 の中央は 292 Hz でリツと夏目の間 |
| 未知曲への汎化（仮説） | 明瞭度で動いたのは「**その曲を学習で見たか**」だけでした |
| **話者を足すこと自体の効果は期待できない（実測）** | **話者の既知性では明瞭度は動きませんでした**（未知話者 +7.3〜+11.1 点 対 既知話者 +8.9 点）。この実測を根拠に base 作り直し（案 B）を一度中止しています |
| 商用リリースには効かない（確認済み） | 上記 |

#### 実測（2026-09-19。shard の先頭だけを range 取得。200 GB は展開していません）

**確認済み:** `huggingface_hub` の `hf_hub_url` + HTTP Range で先頭を取り、tar のヘッダを
自前で辿って中身を読みました（`vocal-only` は gated ですが、利用者の HF token
（`ayousanz`）でアクセスできます）。**全 200 GB を落とさずに素材の質を測れます。**

| 対象 | 実測 |
|---|---|
| `train-0000.tar`（MP3 側）13 本 | **全部 44,100 Hz / stereo**。長さ 169〜285 秒（**曲まるごと**）。`dataset_category` は `official` と `cover` の両方 |
| 実効帯域（MP3 13 本） | **p05 15,462 / p50 15,781 / p95 16,091 Hz。16,000 Hz 以上は 7.7%（1/13）** |
| peak（MP3 13 本） | **11/13 が 1.0 超**（最大 1.257）。配信マスタリングそのもの |
| `vocals-0000.tar` の 1 本目 | `--0fvYUtDas.wav` = **44,100 Hz / stereo / PCM_16 / 198.5 秒 / 35 MB**。**MP3 ではなく WAV** |
| 同・レベル | **peak 0.759 / RMS 0.108 / clipping 0%**（分離で peak が下がる） |
| 同・実効帯域 | **15,781 Hz**（MP3 側の p50 と一致 = **元の MP3 の lowpass を引き継いでいる**） |
| 同・無音 | フレーム energy が −60 dB 未満の割合 **17.4%**（イントロ・間奏を含むため） |
| 同・F0（RMVPE、CPU、全長） | **有声率 69.2%、F0 p05/p50/p95 = 192 / 284 / 522 Hz**、音域 32.8〜39.2 半音 |

**確認済み: 帯域は mel 契約に対して足りています（心配は解消）。** `mel` は
40–16,000 Hz / 128 bin なので、**最上位 bin の中心は 15,540 Hz** です
（`librosa.mel_frequencies(n_mels=130, fmin=40, fmax=16000)` で確認）。実測の p50 15,781 Hz は
**その上**にあり、**中心が実効帯域より上にある bin は 0/128**（p05 の 15,462 Hz でも 1 本だけ）。
**JVS-MuSiC（24 kHz、12–16 kHz が丸ごと空）とは規模が違います。**

**注意: ただし既定の検査を有効にすると 92% が弾かれます。** `AuditThresholds(min_bandwidth_hz=16000.0)`
は sample rate から決めた値で、**mel 契約（上位 bin の中心 15,540 Hz）から決めた値ではありません**。
この素材に使うなら **`min_bandwidth_hz` は 15,400 Hz 前後**に校正します。**閾値を素材に合わせて
緩めるのではなく、mel の上位 bin の中心という根拠で決めること。**

**訂正: 話者ラベルはあります（前の版の「無い」は誤りでした）。** sidecar JSON の
キーは `id` / `url` / `title` / **`channel_id`** / **`channel_title`** / `duration_sec` /
`published` / `view_count` / `like_count` / `category_id` / `source_playlist` /
`topic_categories` / `language` / **`dataset_category`** / `dataset_genre` /
**`dataset_score`** です。**`channel_id` を第一近似の話者 id として使えます。**

**確認済み: ただし `official` では channel = 話者になりません。** 実測 13 本の
`channel_title` には **`Warner Music Japan`** が含まれます（レーベルの公式チャンネル =
複数アーティスト）。一方 `cover` 側は個人の歌い手チャンネル（`そらる / soraru`、`Misumi`、
`シクフォニ` など）で、**1 channel = 1 話者として扱えます**。
**決定として置くべき方針: 話者条件を効かせる base には `dataset_category == "cover"` を使い、
`official` は channel を話者 id に使わないこと。** `official` も使うなら ECAPA での
話者クラスタリングが必要です（**未実装**）。

#### 残る課題

1. **話者数が増えたときの eval。** `eval_items` は**話者ごとの本数**です。23 話者で 69 サンプル・
   10 分超かかっているので、数百話者では eval が破綻します（`eval_items: 1` と大きい
   `eval_interval` が必須。[CLAUDE.md](../CLAUDE.md)）。`spk_bank` は
   `nn.Embedding(n_speakers, 32)` なので**重み自体は 1 万話者でも 1.3 MB** で問題ありません。
2. **cover による曲の重複。** `cover` が 3/13 あり、**同じ曲が別の動画として複数入ります**。
   曲単位 split（`_song_of()`）は動画 ID を曲名にすると効かず、**leakage します**。
   `title` での曲名照合が要ります（**未実装**）。曲名は casefold して表記ゆれを畳む既存の
   規約に合わせること。
3. **低音域は埋まりません（2026-09-19、n=75 で確定）。** C3 未満の滞在率の中央値は
   **1.224%** で、夏目悠李の 3.6% に届きませんでした（[実行計画](svc-plan.md) 12 節 P0-2）。
   カテゴリ別（cover 2.05% / official 1.10% / diverse 0.46%）でも同じです。
   **「YouTube なら低音男声が入る」は成り立ちません。** ただし**低い曲が無いわけではなく**、
   21/75 が 3.6% を超え最大 61.4% でした。**低い部分集合を選り分けられるかは別の問い**で、
   まだ事前登録していません。**そして低い曲は `diverse` と `official` に偏る**ので、
   **低音を取りに行くと話者ラベルを失います。**
4. **上限が既存素材と別物です（2026-09-19 実測、決定的）。** GT mel を NHVSing に通した
   再合成の mel L1 は **0.5279** で、既存 5 コーパスの範囲（0.2929〜0.3371）の **+56.6%**
   外側でした（12 clip・7b 節と同一条件。Mann-Whitney U で p = 0.000002、
   既存 60 clip の最大を 10/12 が上回る）。F0 半音誤差と V/UV 一致率も同じ向きです。
   **原因は分離アーティファクトだけに帰属できません** —— マスタリング済み配信音源・
   ボーカル分離・曲まるごと、の 3 つが同時に違うためです。**この素材で測った「上限との差」は
   既存の数値と並べられません。**（[実行計画](svc-plan.md) 12 節 P0-3）

**次の測定（1 shard 全体。それでも 5 GB）:**

```bash
uv run python -m preprocess.svc.run --wav-dir download/tts_vocal/shard0000 \
  --out data/probe_tts --song-parts 1 --limit 200 --device cuda
```

`--limit` で先頭 200 ファイルだけを通し、audit の除外内訳（`silence` / `clipping` /
`band_limited`）・実効帯域・有声率・F0 分布を**分布として**見ます。上の実測は
**帯域 n=13 / F0 n=1** なので、**話者の音域分布と分離品質はまだ 1 本ぶんの証拠しかありません。**

**注意: 上限（`*_vocoder_only.wav`）の意味が変わります。** 分離 stem を GT mel にすると、
**上限自体が分離アーティファクトを含みます。** M5 の客観指標はすべて「上限との差」で読む
設計なので（[評価計画](svc-evaluation.md)）、**この素材で作った上限と、リツ・VocalSet で
作った上限は混ぜられません。**

**注意: loudness の分布が配信側へ寄ります。** 配信用に整えられた音源は peak 1.0 付近で、
学習素材（波音リツ DB は peak 0.107）から大きく外れます（[CLAUDE.md](../CLAUDE.md)）。
**1,000 時間の大半がそちら側なら、学習分布そのものが動きます。** 副作用として**持ち込み音源に
強くなる**（`--match-loudness` が要らなくなる方向）可能性と、**リツのような素の DB が
out-of-distribution になる**可能性の両方があります。**どちらも未検証です。**

## 5. どれだけ必要か

**見積もり:** 部品ごとに「必要量」と「4 節で確保できる量」を並べます。必要量の根拠は
このリポジトリの実測（[データと計算資源](svc-data-compute.md) 2 節・3 節、
[実行計画](svc-plan.md) M3–M5）です。

| 部品 | 必要量 | 根拠 | 4 節で確保できるか |
|---|---|---|:-:|
| ボコーダー再学習 | **20〜50 h / 多話者 / 44.1 kHz 以上** | NHVSing V3 は 10 コーパス（時間は非公表）。一般的な neural vocoder の規模 | **可**（歌唱 39 h + speech） |
| SVC base（現状再現） | **18〜20 h / 20 話者以上** | 現 base が 23 話者・約 18 h で M5 まで到達 | **可**（ぎりぎり） |
| SVC base（品質を上げる） | **100〜300 h / 20〜50 人** | [データと計算資源](svc-data-compute.md) 3 節の本学習案 | **不可**（39 h しかない） |
| target singer | **5〜10 h** | 波音リツ 10.41 h で M4 が機能した実測 | **可**（波音リツ） |
| 未知 source の評価 | 10〜20 clip / 学習に出ない歌手 | M5 の test set 構成 | **可**（VocalSet を歌手分割） |

**確認済み: 「話者を足す」の期待値は低いと実測が言っています。** 明瞭度で動いたのは
**その曲を学習で見たかどうかだけ**で、**話者の既知性・`num_steps`・GAN・fine-tune の step 数・
base の学習量は動きませんでした**（[CLAUDE.md](../CLAUDE.md)、[実行計画](svc-plan.md) M5）。
この実測を根拠に **base 作り直し（案 B）を一度中止**しています。

**したがって商用化の作業は「品質向上」ではなく「同じ品質を権利クリーンに作り直す」ことが
主目的になります。** 品質を上げたいなら、素材の量より**曲の多様性**（同一歌手でも未知曲で
落ちる）を狙うほうが実測と整合します。**仮説であり、未検証です。**

**注意: この規模では +12.7 点（明瞭度の上限との差）が水準という可能性が残っています。**
「データを増やせば上がる」は確認済みではありません。

## 6. 不足を埋める 4 つの手段

| 手段 | 得られるもの | 費用（見積もり） | 主な障害 |
|---|---|---|---|
| **A. 自前収録** | 権利が最も明確。target も base も作れる | **1 歌手 5 h で ¥10 万〜35 万＋歌唱料・権利買い取り** | 歌唱料と権利買い取りの相場が不明。楽曲側の権利も要る |
| **B. 有償ライセンス** | 時間を金で買える | **low〜mid six figures USD**（二次情報） | 桁が個人開発の範囲外 |
| **C. speech で話者数を稼ぐ** | 話者数 328（VCTK + AISHELL-3）と低音域 | **ほぼ 0** | 歌唱ではない。効果は未検証 |
| **D. augmentation で擬似話者** | 少数の権利クリーン歌手から話者数を増やす | **ほぼ 0**（計算費のみ） | 未実装。擬似話者が実話者の代わりになるかは未検証 |
| **E. 手元の 1,000 h 素材**（4.5 節） | 日本語 1,000 h / 低音域の男声 | **ほぼ 0** | **商用トラックには使えません**（CC BY-NC 4.0）。話者ラベル無し・cover 重複・分離 stem の 4 点が未解決 |

**A の内訳（見積もり）:** usable な歌唱 1 時間には、テイク・休憩・セットアップを含めて
**スタジオ 3〜4 時間**を見ます。実勢はスタジオ **¥3,000〜10,000/h**、エンジニア
**¥15,000〜50,000/日**、同人の歌唱依頼が **1 曲 ¥10,000 前後**です。1 歌手 5 h usable なら
スタジオ 15〜20 h で **¥4.5 万〜20 万**、エンジニア 2〜3 日で **¥3 万〜15 万**。
**10 歌手 50 時間ならスタジオとエンジニアだけで ¥75 万〜350 万**です。

**確認済みでないこと: AI 学習用の権利買い取り（実演家の権利を含む）の相場は、今回の調査では
見つかりませんでした。** 上の金額は**収録の実費だけ**で、**権利の対価を含みません**。
桁が変わり得ます。

**A の隠れた要件: 楽曲の権利も要ります。** 「誰が歌うか」と「何を歌うか」は別です。
**PD の童謡・クラシック**か**自作曲**にすると楽曲側が消えます。PJS はこの方式（第一著者が
音素バランスを保って作曲）で、**再現可能な研究用素材を作るための設計**です。

**D について:** SVC では online の `pitch_aug` が使えません（特徴量が事前計算済みのため
`train.py` が明示的に SystemExit します）。**augmentation は特徴量抽出の前**に当てる必要が
あります（[CLAUDE.md](../CLAUDE.md)）。formant / pitch を振って擬似話者を作る案はこの制約の
内側に収まりますが、**未実装**です。

## 7. 費用構造: データが支配し、GPU は誤差

**確認済み（実測）:** 計算側は安いことが分かっています。

| 工程 | 実測 | 出典 |
|---|---|---|
| base 学習（23 話者・18 h） | 8.14 step/s、30,000 step で約 61 分、peak VRAM 1.95 GB。**継続 run の実費 $2.148**（GPU $1.157 / download $0.868 / storage $0.111） | [データと計算資源](svc-data-compute.md) 8 節 |
| 特徴抽出 | 25 shard を 52.3 分（**約 19 倍速**）。GPU 使用率 6% | 同 |
| shard の容量 | **約 1.6 GB / audio-hour** | 同 |
| target fine-tune（7.6 h・20,000 step） | 82 分、実費約 $0.75 | 同 |

**見積もり: 100 時間の権利クリーン corpus でも、抽出約 5 時間・学習数十ドルです。**
一方 6 節の収録費は**百万円単位**です。**したがって商用化の意思決定は、GPU 予算ではなく
データ取得予算の話になります。**

**注意: 通信量が実費の 40% を占めました。** 100 時間なら shard が約 160 GB になるので、
**回収の設計（必要な checkpoint だけ落とす）を先に決めること**（同 8 節）。

## 8. 段階計画と事前登録する判定条件

**推奨:** 素材の取得より**権利の確定が先**です。S0 は無料で、結果によって S1 以降の内容が
変わります。

### S0. 権利の確定（費用 0、データ取得なし）

| やること | 誰に | 何を確定するか |
|---|---|---|
| 波音リツの機械学習条項 | 権利者（canon-voice） | 規約は商用可・再配布可だが**機械学習が記載なし**。学習と重み配布の可否を書面で |
| No.7 の商用条件 | voiceseven.com | 「商用はご相談」なので**交渉の余地**。日本語 44.1 kHz 以上の貴重な素材 |
| RMVPE 重みの由来 | 配布元 / 論文著者 | MIT 宣言の根拠と学習素材。**同梱するなら必須** |
| SingVERSE の clean 側の由来 | amphion | CC BY 4.0 の宣言が上流と整合するか。**18.14 h は全体の半分近く** |
| CC BY-SA を base に入れるか | **法務判断** | ShareAlike が学習済み重みに及ぶか。及ぶなら PJS も GTSinger と同じ問題になる |
| `tts-dataset/*` を**商用配布**に使えるか | **法務判断** | 学習の可否（利用者は「問題ない」と判断）とは**別の層**。原盤・実演・楽曲の許諾があるかで 4.1 節の集計が変わる（4.5 節） |

**問い合わせ文面のたたき台は [台帳](svc-dataset-ledger.md) 10 節に 6 件あります**（御丹宮くるみ /
夏目悠李 / **波音リツの機械学習条項** / **No.7 の商用** / **RMVPE 重みの由来** /
**SingVERSE の素材由来**。後ろの 4 件は 2026-09-19 に追加）。**送った日付と回答は台帳 1 節の
表へ追記してください** —— この 4 件が閉じるまで商用トラックは動きません。

### S1. 権利クリーンな base を作り、現行 base と比べる

**確認済みの手順で回せます。** `preprocess.svc.run` は WAV ディレクトリから shard を作り、
再実行で bit 一致します。話者ごとにディレクトリを分け、`--song-parts` を渡すこと。

```bash
uv run python -m preprocess.svc.run --wav-dir download/vocalset/<歌手> \
  --out data/clean_<歌手> --song-parts 1 --max-hours 0.75
uv run python -m train --config configs/svc_base.yaml \
  --data_dirs data/clean_* --run_name svc_base_clean_01 --out_root log --device cuda
```

**事前登録する判定条件（規則を先に書く）:**

| 指標 | 比較 | 合格条件 |
|---|---|---|
| 明瞭度（CER の上限との差） | 現行 base と同じ hold-out clip | **差の中央値**で比べ、**対応のある検定で有意差なし**なら同等と判定 |
| 話者類似度（回復率） | 同上 | 同上 |
| 内容 cos | 未知 source | 現行 base から 0.02 を超えて落ちない（M4 の規則を流用） |

**注意: 中央値だけを見る規則は壊れ方を通します。** 実測で clip 単位の振れ幅は最大 73.6 点
あり、中央値で 2.3 点の「改善」が出ても clip 単位では 2 勝 2 敗 2 分（p = 1.0）でした
（[CLAUDE.md](../CLAUDE.md)）。**規則に対応のある検定を入れること。**

**注意: 上限を共有できない指標は系をまたいで比べられません。** CER・信号品質・明るさは
「GT mel を**自分の**ボコーダーに通した再合成」を基準にします。**S2 でボコーダーを変えたら、
S1 の数値とは比べられません。**

### S2. ボコーダーを権利クリーンにする（V-a: NHVSing 再学習）

3 節の V-a を採ります。**mel 契約を維持できるので音響モデルは再学習不要**ですが、
**上限が変わるので README と doc の「上限との差」は全部測り直し**になります。
V3.1 → V3.2 の据え置きを決めたときと同じ理由です。

**やることの順序:** ①権利クリーン素材でボコーダーを学習 → ②上限（`*_vocoder_only.wav`）を
作り直す → ③S1 の全指標を再測定 → ④**旧値と新値を並べて**文書を更新（**黙って差し替えない**）。

#### 調査結果（2026-09-20。上流リポジトリを一次資料として確認）

**確認済み: 道具の面では可能です。学習コードとレシピが公開されています。**

| 項目 | 確認した内容 |
|---|---|
| コード | **MIT**。`train_v3.py`（76 KB）・`preprocess.py`・`dataset.py`・`discriminator.py` が公開 |
| レシピ | **配布重みそのもののレシピが公開**（`config_v3_1.yaml` の冒頭が「V3.1 配布重み(exported_models/v3_1)の学習レシピ」と明記） |
| **mel 契約** | **完全一致**: 44,100 / hop 256 / fft 2048 / win 2048 / 128 mel / 40–16,000 Hz、`mel_format: diffsinger`。**V-a が mel 契約を保てることがコードで裏づけられました** |
| 前処理 | `preprocess.py --indir <wav dir> --out <npz dir> --config <cfg>`。**再帰 glob（`**/*.wav`）で、sr が違えば自動 resample** —— 我々の 44.1 / 48 kHz 混在素材をそのまま渡せます。**train/test は自動で分けない**（2 回実行する） |
| 学習 | `train_v3.py --config <cfg>` の 1 本。**単一 GPU 前提**（「複数のGPUの使用を想定していません」と明記） |
| 主要ハイパー | `batch_size 256` / `crop_frames 64`（372 ms のランダムクロップ）/ `n_epoch 1021` / `adversarial_start 100` / `adversarial_end 1000`（**最後の 20 epoch は D を止める「縞消し」フェーズ**）/ `lr_g 4e-4`・`lr_d 2e-4` / `pitch_aug_prob 0.5` |
| 判別器 | MPD（周期 3/5/7/11/17/23/37）+ MRD。MSD は使わない |
| F0 | **RMVPE**（40–1,200 Hz）。**我々と同じ抽出器**なので、8 節 S0 の RMVPE 由来確認がここにも効きます |
| 正規化 | `target_rms: 0.111`（学習データの RMS 正規化基準） |
| 依存 | torch / torchaudio / librosa / **pyworld** / onnx など。**`pyworld` は当リポジトリの `pyproject.toml` に無い**ので追加が要ります |

**見積もり（要実測）:** 商用可素材 39 h を `cut_wavs.max_dur 30.0` で切ると **約 4,700 segment**、
`batch_size 256` なら **1 epoch ≈ 18 step**、`n_epoch 1021` で **約 18,400 step**。桁としては
我々の SVC base（60,000 step / GPU 約 2 時間 / 実費 $2.148）と同程度ですが、**1 step の重さが
違います**（波形生成 + MPD 7 期間 + MRD）。**「1 日以内・$10 未満」を出発点の見積もりとし、
最初の 100 / 1,000 step で測り直します。**

**確認していないこと（上流に記載がありません）:** 学習時間・VRAM・**必要なデータ量**。
V3 は **10 コーパス**で学習しており、**39 h（SingVERSE を除くと 21 h）で足りるかは未知**です。
上流は「V1 は単一話者特化で**学習話者以外ではアーティファクトが出やすい**、V2/V3 は多話者で
汎用」と説明しているので、**話者数が少ないと V1 の弱点へ戻る危険**があります。
**これが商用トラック最大の技術リスク**です。

**未検証の経路なので、借りる前に 1 step 踏むこと。** `train_v3.py` が 1 step 回り、
TensorBoard に `d_loss` / `adv` が出ることを確認してから vast.ai を借ります
（`leapsinger-experiment` の「未検証の学習経路は、借りる前に smoke で踏む」）。
**上流リポジトリでの実行は当リポジトリの hook 対象外**なので、どこで踏むかは利用者の判断です。

### S3. 追加データで品質を上げる（条件つき）

**起動条件:** S1 で同等が確認でき、かつ 6 節の A か B の予算が付いたとき。
**起動しない条件:** 5 節の実測（話者を足しても明瞭度は動かない）が覆らないうち。
**この順序を守ること** —— 実験の後に規則を決めると、train loss で checkpoint を選ぶのと
同じ間違いをします（[実行計画](svc-plan.md) M4）。

## 9. 残るリスクと要ユーザー判断

**要ユーザー判断: 再配布物の「CC BY 宣言」は上流の適法性を保証しません。**
SingVERSE・Emilia-YODAS・RMVPE の重みは、いずれも**再配布者が permissive を宣言**して
いますが、**素材の由来は書かれていません**。商用で同梱するかは、この不確かさを受けるかの
判断です。**「宣言があるから確認済み」と書かないこと。**

**要ユーザー判断: ShareAlike が学習済み重みに及ぶか。** 条文からは決まりません
（[先行研究・ライセンス・リスク](svc-prior-art-license.md) 3 節）。及ぶと判断するなら、
**CC BY-SA 4.0 の PJS も base に入れられません**（影響は 1 時間未満なので小さい）。
**及ばないと判断しても、GTSinger の NC は別途効きます。**

**確認済み: 同意（consent）はライセンスとは別に必要です。** GTSinger の README は
「本人の同意なく特定個人の歌声を生成すること」を明示的に禁じています。**SVC はまさにその
能力**なので、**変換先の歌手の同意が前提**です。素材のライセンスが商用可でも、
**実演家の同意が要る**という構造は変わりません。

**要ユーザー判断（日本法）:** 著作権法 **30 条の 4**（情報解析など非享受目的の利用）が
学習段階の複製を許す枠組みですが、文化庁の
[「AIと著作権に関する考え方について」](https://www.bunka.go.jp/seisaku/bunkashingikai/chosakuken/pdf/94037901_01.pdf)
（2026-09-19 取得。文化審議会著作権分科会法制度小委員会、2024-03-15）は、
**「特定のクリエイターの作品の集中的な学習」**や**享受目的が併存する場合**を同条の適用外に
なり得るものとして挙げています。**target fine-tune はまさに特定歌手の集中的な学習**です。
**また、規約（契約）の制約は権利制限規定では解除されません** —— 「30 条の 4 があるから
規約を無視できる」とはなりません。**法務の判断が要ります。**

**確認済みでないこと: 声そのものを直接規律する明文の規定は、今回の調査では見つかりません
でした。** パブリシティ権・人格的利益・不正競争防止法の議論としては扱われますが、
**この調査の範囲では結論を出せません。要ユーザー判断です。**

**要ユーザー判断: 帰属表示の運用。** CC BY / CC BY-SA / Apache-2.0 の素材を使うと、
**配布物に帰属表示が必要**です。`LICENSE` の NOTICE と配布物の `CREDITS.txt` に、
**素材ごとの表記・URL・取得日**を残す運用を決めてください（ccMixter のような
1 曲ずつライセンスが違う素材を使うと、**表記の管理そのものが作業になります**）。

## 10. 出典

**一次資料（2026-09-19 取得。特記したものを除く）:**

| 対象 | URL |
|---|---|
| NHVSing（コード MIT / 重みは非商用） | https://github.com/wavtechyukky/NHVSing |
| BigVGAN v2 44 kHz 128 band 256x（MIT） | https://huggingface.co/nvidia/bigvgan_v2_44khz_128band_256x |
| ContentVec（MIT。LibriSpeech 960 h） | https://huggingface.co/lengyue233/content-vec-best / https://arxiv.org/abs/2204.09224 |
| RMVPE 重みの配布元（MIT 宣言） | https://huggingface.co/lj1995/VoiceConversionWebUI |
| RMVPE 参照実装（Apache-2.0） | https://github.com/Dream-High/RMVPE |
| VocalSet（CC BY 4.0） | https://zenodo.org/records/1442513 |
| SingVERSE（CC BY 4.0、18.14 h、44.1 kHz） | https://huggingface.co/datasets/amphion/SingVERSE |
| 波音リツ 利用規約（商用可・再配布可） | https://www.canon-voice.com/terms/ |
| PJS（CC BY-SA 4.0、48 kHz） | https://sites.google.com/site/shinnosuketakamichi/research-topics/pjs_corpus |
| NIT-SONG070-F001（CC BY 3.0 の記載） | https://sinsy.sourceforge.net/readme_hts_voice_nitech_jp_song070_f001.php |
| JSUT-song（非商用。商用は要相談・再配布不可） | https://sites.google.com/site/shinnosuketakamichi/publication/jsut-song |
| No.7 / voiceseven（商用は要相談） | https://voiceseven.com/ |
| VCTK 0.92（CC BY 4.0、48 kHz、110 話者） | https://datashare.ed.ac.uk/handle/10283/3443 |
| AISHELL-3（Apache-2.0、218 話者、85 h） | https://huggingface.co/datasets/AISHELL/AISHELL-3 |
| Emilia / Emilia-YODAS（CC BY-NC 4.0 / **CC BY 4.0**） | https://huggingface.co/datasets/amphion/Emilia-Dataset |
| Children's Song Dataset（CC BY-NC-SA 4.0） | https://zenodo.org/records/4785016 |
| SingStyle111（**論文 PDF の record**。4.4 節） | https://zenodo.org/records/10265401 |
| `tts-dataset/japanese-singing-voice`（CC BY-NC 4.0、約 1,000 h、MP3、YouTube 由来） | https://huggingface.co/datasets/tts-dataset/japanese-singing-voice |
| `tts-dataset/japanese-singing-voice-vocal-only`（CC BY-NC 4.0、約 199.6 GB、gated） | https://huggingface.co/datasets/tts-dataset/japanese-singing-voice-vocal-only |
| SingNet（3,000 h。音声は非公開） | https://arxiv.org/abs/2505.09325 |
| CP Singing Voice Dataset（Ircam Forum License） | https://zenodo.org/records/18773342 |
| Seed-VC の学習データ（Emilia-101k = 約 101,000 h の speech） | https://arxiv.org/html/2411.09943v1 |
| 文化庁「AIと著作権に関する考え方について」（2024-03-15） | https://www.bunka.go.jp/seisaku/bunkashingikai/chosakuken/pdf/94037901_01.pdf |

**二次情報（検証していません）:**

| 対象 | URL |
|---|---|
| 商用歌唱データのライセンス価格帯・「オープンな clean 歌唱は約 230 h」 | https://thevocalmarket.com/blogs/enterprise/state-of-vocal-data-licensing-2026 |
| 歌唱依頼・レコーディングの実勢価格（同人相場） | https://ut-9.net/making/doujin_kasyou_souba.html / https://music-s.com/recording-costs/ |

**ローカル根拠:** 時間・sample rate・実効帯域の実測は
[データセット台帳](svc-dataset-ledger.md) 4b 節、学習費用の実測は
[データと計算資源](svc-data-compute.md) 8 節、明瞭度と話者類似度の実測は
[実行計画](svc-plan.md) M4–M5 と [CLAUDE.md](../CLAUDE.md)。

**比較の目安（確認済み）:** blind preference で負けた Seed-VC は **Emilia-101k（約 101,000
時間の speech）**で学習されています。**我々の base は 18 時間**です。**この差を公開素材で
埋めることはできません** —— したがって競合の指標は「データ量」ではなく、
**F0 追従・V/UV・timing のように我々が上回っている軸**に置くのが妥当です
（[実行計画](svc-plan.md) M5）。
