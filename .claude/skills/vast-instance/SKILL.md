---
name: vast-instance
description: vast.ai の Linux GPU インスタンスを借りて学習を回し、成果物を回収して破棄するまでの手順。GPU が必要になったとき、インスタンスを探す・作る・接続する・止めるとき、リモートで学習を走らせるときに使う。時間課金なので、作成と破棄の判断規則を含む。
---

# vast.ai で学習を回す

**手元の Windows 機は開発・推論・検証用。学習は vast.ai の Linux インスタンスで行う。**
時間課金なので、借りている時間が短くなるように順序を決める。

## 0. 準備（初回のみ）

```bash
uv sync --extra ops        # vastai CLI が入る
```

API token は `.env`（`.gitignore` 対象）に置く。`tools/vast.py` が
`VASTAI` / `VAST` / `VAST_API_KEY` / `VASTAI_API_KEY` / `VAST_AI_API_KEY` / `VAST_TOKEN` /
`VASTAI_TOKEN` のどれかを自動で拾う。**token を表示・コピー・コミットしない。**
見つからないときは `.env` にある**変数名だけ**（値は出さず）を表示して止まる。

## 1. 借りる前に手元で済ませる

インスタンスは起動した瞬間から課金される。次は**借りる前に**終わらせておく。

- コードを `origin/main` に push しておく（bootstrap が clone する）
- config を決めておく（[leapsinger-experiment](../leapsinger-experiment/SKILL.md) の 1〜2）
- データの転送手段を決めておく（shard をどうやってインスタンスへ置くか）
- 手元で `uv run python tools/smoke/run_smoke.py --device cpu` を通しておく
- **必要になるスクリプトを書き切っておく。** M3 では課金中に実装を書きたくなり、TDD の順序を
  1 度崩した。素材の構造（GTSinger の階層など）は**借りる前に手元のサンプルで確かめられる**。
- **ディスク量を見積もっておく。** ディスクは実効料金に効く（下の 3 節）。M3 の実測は
  音声 11 GB + shard 29 GB + 環境 11 GB = **51 GB**。
- **通信量の単価を見ておく。** `search` の `$/TB` 列。**環境構築だけで 8 GB 前後**落ちるので
  どの run にも必ず載る。実測で offer 間に **13 倍の開き**（0.3〜4.0 $/TB）があった。

## 2. 探す（課金なし）

```bash
uv run python tools/vast.py search --vram 24 --max-price 0.60
```

既定の絞り込み: `num_gpus=1` / `reliability > 0.98` / `verified=true` /
`cuda_max_good >= 13.0`（cu130 wheel を使うため）/ `disk_space >= 60` / 安い順。

VRAM の目安は [`doc/svc-data-compute.md`](../../../doc/svc-data-compute.md)。
**ただしその表は計画値です。** multi-singer base の実測 peak は **1.95 GB** で、
24 GB 級は要りませんでした（3 節）。`max_batch_size` を上げる・GAN を足す・crop を伸ばす
ときだけ大きい VRAM が要ります。**まず実測を見て、無ければ小さめで試す。**

**ディスクは 40 GB 以上。** torch cu130 + nvidia 系 wheel だけで 8 GB 前後を使う。

## 3. 作る（ここから課金）

```bash
uv run python tools/vast.py create <offer_id> --disk 60 --yes
```

- `--yes` が無ければ実行されない。ただし**料金プレビューは当てにならない**（3b 節。`id=` 検索が
  空を返す）。**検索一覧に出ている `$/hr` を価格の根拠にする。**
- 既定イメージは `vastai/base-image:cuda-13.0.3-auto`（ホスト CUDA 13.0 系＝cu130 wheel と一致、
  gcc 入りなので Linux では `torch.compile` が効く）。
- `--rmvpe` を付けると前処理用の RMVPE 重み（181 MB）も落とす。
- onstart で `tools/vast_bootstrap.sh` が走る。uv 導入 → clone → `uv sync` → CUDA 疎通 →
  **倍音和の compile 経路が効いているかの確認** → 単体テスト（3 ファイル全件）、まで自動。
- **`--disk` は料金に直接効く。** M3 で 150 GB を付けたら $0.136/hr の offer が実効 **$0.21/hr**
  になった（+54%）。必要量を 1 節で見積もってから決める。
- **VRAM は控えめでよいことが多い。** multi-singer base の実測 peak は **1.95 GB** で、
  24 GB 級は要らなかった（`max_batch_size` が先に効くため）。VRAM より
  **回線速度（`down`）とディスク**で選ぶほうが効く場面がある。

## 3b. 実運用で分かったこと（2026-08-30、M2 で一通り回した）

**確認済み:** 次はすべて実際に踏んだものです。

| 事象 | 対処 |
|---|---|
| `vastai execute <id> '<cmd>'` は**制限付き**で、任意コマンドは `Invalid command given` (400) | **SSH を使う。** `execute` は当てにしない |
| SSH には**鍵の登録**が要る | `vastai show ssh-keys` で確認。無ければ `vastai create ssh-key`。**アカウントに登録済みの鍵が手元の鍵とは限りません**（実測: 登録は別マシンの鍵で、手元の `~/.ssh/id_ed25519_vast` では `Permission denied (publickey)` になった）。その場合は**インスタンスへ個別に付ける**: `uv run python tools/vast.py attach <instance_id>`（既定 `~/.ssh/id_ed25519_vast.pub`。秘密鍵を渡すと止まる）。数秒で有効になる。**作成直後に済ませること**（M4 で 15 分を無駄にした） |
| `vastai destroy instance` は確認プロンプトを出し、stdin が無いと `Aborted.` | `-y` が要る。`tools/vast.py` が渡すようにした |
| `search offers 'id=<N>'` が、その offer が実在しても**空を返す** | `create` の料金プレビューは当てにならない。**一覧に出ている価格を見る** |
| offer ID の**回転が速い** | 検索してすぐ作る。数分置くと消える |
| インスタンスの `logs` には bootstrap の出力が出ない | onstart は `/root/bootstrap.log` へ落としてある。SSH で見る |
| アカウントに**自分が作っていないインスタンス**が居ることがある | `label` と `image` で見分ける。**自分のもの以外は触らない** |
| Windows から書いたスクリプトを `bash -s` で流すと `set: pipefail: invalid option name` | **CRLF が混ざっている。** Python で書くなら `write_text` に `newline="\n"` を渡す |
| `set -euo pipefail` のスクリプトが nvidia-smi の直後に無言で死ぬ | `cmd \| tee f \| head` の **SIGPIPE**。`head` が先に閉じると `pipefail` + `set -e` で run 全体が落ちる。**先頭数行だけ見る用途で `head` をパイプの末尾に置かない** |
| `show instances` が日本語 Windows で `'cp932' codec can't encode character` で落ちる | `tools/vast.py` が出力を UTF-8 で受けてから安全に表示するようにした |
| 突然どのコマンドも `ImportError: DLL load failed while importing _socket` で落ちる | **uv の shim が壊れた Python を参照している。** `uv tool install --reinstall` では直らない（壊れているのは shim の側）。`tools/vast.py` はツール環境の Python から `-m vastai.cli.main` へ落ちるようにした |
| その fallback も同じエラーで落ちる | **`uv run` が `PYTHONHOME` を子へ継承している。** ツール環境の Python が親の標準ライブラリを読みに行く。`_child_env()` が親を指す変数だけ落とす |
| アカウントに**他の実験のインスタンス**が動いている | `label` で見分ける（実測: 自分の SVC 実験と、別プロジェクトの Beatrice 学習が同居していた）。**自分のもの以外は触らない** |
| `create` の応答に **`instance_api_key` が平文で出る** | `tools/vast.py` が伏せるようにした。**ログにもチャットにも残さない** |

### 3c. M3 で追加で踏んだこと（2026-08-30）

| 事象 | 対処 |
|---|---|
| **リモートで `pkill -f "..."` を打つと自分の SSH シェルごと死ぬ** | コマンド文字列自体がパターンに一致する。`pkill -9 -f "python3 -m trai[n]"` のように**角括弧で自己一致を外す**。**hook が止めるようになりました** |
| `ssh ... 'cmd &'` は SSH が channel を閉じないので戻ってこない | `setsid nohup ... < /dev/null > log 2>&1 &` で完全に切り離し、ログを別途 `tail` する |
| 長時間ジョブの進捗を `ssh` で毎回取ると turn を食う | 完了マーカー（`echo "=== 完了 ==="`）を仕込み、`until grep -q ...; do sleep 60; done` を**バックグラウンドで 1 本**回して通知を待つ |
| ダウンロードの進捗バーが**巨大な出力**になる | 取得するときは `grep -aE "^\[...\]"` などで必ず絞る。生の `tail` を投げない |
| **素材はインスタンスと一緒に消える** | M3 の shard 29 GB は `destroy` で消えた。学習を継続するには**素材の再生成（実測 約 65 分）と checkpoint の再アップロードが要る**。「学習だけ 1 時間」で見積もると外す |
| HF の取得速度は回線ではなく**先方の律速**で決まる | 7.4 Gbps の offer でも 7,977 ファイルに 20 分以上かかった（前回は 900 Mbps の offer で 9 分）。回線速度で選んでも縮まないことがある |

### 3d. P1 で追加で踏んだこと（2026-09-20）

| 事象 | 対処 |
|---|---|
| **`create` の料金プレビューはやはり空**（`offer 51030538: ? x? VRAM 0GB`） | **実効料金は作成後の `instances` に出ます**（今回 offer $0.081 + disk 120 GB で **$0.1133/hr**）。検索一覧の `$/hr` に disk ぶんを足した額だと思っておく |
| bootstrap が clone するのは **`main`** | 作業ブランチで動かすなら、**借りる前に push** して、インスタンス上で `git fetch origin <branch> && git checkout -B <name> origin/<branch>` → **`uv sync` をやり直す**（依存が変わっていることがある） |
| **gated な HF データセットはインスタンスから落とせない** | 手元の token ファイルを **scp で渡す**。**値を表示しないこと**: `scp ~/.cache/huggingface/token root@host:/root/.cache/huggingface/token` の後に `chmod 600`。`huggingface_hub.get_token()` がこの場所を読む |
| 比較相手の checkpoint が**手元に無いと A/B が組めない** | **借りる前に確かめる。** 今回は `.m0data/m3c/ckpt_060000.pt`（141 MB）と `out/m5/where_base60000/`（上限つき 6 clip）が残っていたので成立した。**残っていなければ比較相手も学習し直しになり、費用が倍**になる |
| 素材の取得は**GPU を使わない**のに課金は同じ | 取得と抽出を**並行させる**。今回は cover の streaming 中に別セッションでリツを取得した |

### 3e. S1 で踏んだこと（2026-09-21）。**どちらも実費になった**

| 事象 | 対処 |
|---|---|
| **`create` が `"success": false` を返したのに、インスタンスは作られていた** | 実測で contract `51816074` が **6 時間課金**された（$0.70）。**`success` を信じない。** 作成の後は必ず `instances` を**目で読む**（下の注意も参照） |
| `instances` の表は**色コードで壊れる**ので grep が空振りする | 実測で「0 台」と誤読した。`\| cat -v \| sed 's/\^\[\[[0-9;]*[a-zA-Z]//g'` を通してから読む。**「grep が空 = 0 台」と決めつけない** |
| **インスタンスが offline になり SSH が両方の口で死ぬ** | 実測で 59% まで進んだ学習を失いかけた。**host 側の障害は起きる。** `ssh` の**接続先は途中で変わる**（`ssh6.vast.ai:16144` → 直 IP）ので、`tools/vast.py ssh <id>` で毎回引き直す |
| **中間 checkpoint を 1 つも回収していなかった** | 6 節に「学習中も定期的に退避する」と**書いてあるのに従わなかった**。**5,000 step ごとに落とす**ループを、学習を始めると同時に回すこと |

**`success: false` でも `instances` を確認すること。** これは料金に直結します。

### 3f. 起動待ちが「永遠に準備中」のまま 4 日課金された（2026-09-26）。**これまでで最大の損失**

**何が起きたか（確認済み、インスタンスのログから）:** S1b 用に借りたホストで
`/root/.ssh/authorized_keys` の権限が壊れており、sshd が
**`Authentication refused: bad ownership or modes`** で**最初から鍵を拒否**していました
（ログに**認証成功 0 回・拒否 308 回**）。起動待ちのループは

```bash
until ssh ... 'grep -q 完了 /root/bootstrap.log' 2>/dev/null; do sleep 30; done
```

で、**`2>/dev/null` が認証エラーを握りつぶし、タイムアウトも無かった**ので、失敗を一度も
報告しないまま待ち続けました。**セッションが切れた後もインスタンスだけが残り、
何も走らないまま約 4 日（$0.0794/hr × 約 90 時間 ≈ $7）課金されました** ――
それまでの計算費の合計（約 $2.5）より大きい損失です。

**対処（道具にした）:** `tools/vast.py wait-ssh <id>` を使うこと。自前の `until ssh` ループを書かない。

```bash
uv run python tools/vast.py wait-ssh <instance_id> --timeout 1200
# 成功: "host port" を 1 行出して終了コード 0
# 認証拒否が 4 回続く: 終了コード 2 → そのインスタンスは捨てて別の offer へ
# タイムアウト:        終了コード 3
```

- **「まだ起動していない」（`Connection refused` / `timed out`）と「この先も通らない」
  （`Permission denied`）を分けます。** 鍵の反映待ちの数回は許し、続いたら止めます。
- **待ちには必ず上限を付けます**（既定 20 分）。
- **接続先は毎回 `ssh-url` から引き直します**（途中で変わるため）。
- ホスト側の権限の問題は**外から直せません**（`attach` は `already` を返すだけ）。**捨てて借り直す。**

**一般則: バックグラウンドの待ちは、成功だけでなく失敗の終端でも止まること。**
「沈黙は成功ではない」。**止まらない待ちは、セッションが切れると課金だけを残します。**

**借りる前のチェックに 1 行足す:** **「評価に使う `--spk-id` と hold-out 曲が、新しい素材で
成立するか」**（[leapsinger-experiment](../leapsinger-experiment/SKILL.md) 6d 節）。
**これは手元で確かめられます。** 今回は学習を始める前に気づけましたが、
**気づかなければ 4 時間ぶん課金してから比較できないと分かる**ところでした。

**接続:**

```bash
uv run python tools/vast.py instances          # ssh_host / ssh_port を控える
ssh -i ~/.ssh/id_ed25519_vast -p <port> -o BatchMode=yes root@<host>
ssh -i ~/.ssh/id_ed25519_vast -p <port> root@<host> 'bash -s' < local_script.sh
scp -i ~/.ssh/id_ed25519_vast -P <port> root@<host>:/root/LeapSVC/log/... .
```

**データはインスタンス上で作る。** 転送するより速く、M1 が Linux でも動くことの確認になります。
回線が速い offer を選べば、素材のダウンロードは数十秒で終わります。

```bash
# インスタンス上で
git fetch origin main && git reset --hard origin/main   # 手元の push を反映
uv run python preprocess/download_scripts/download_ritsu.py --voice kire
uv run python -m preprocess.svc.run --wav-dir download/ritsu --out data/x --device cuda
```

**実測（RTX A4000 / $0.098 per hour）:** M1 抽出が 1 曲 42 秒、SVC の overfit 学習が
**13〜15 step/s**、M2 一式で 27 分・**実費およそ $0.04**。bootstrap では
`compiled path active: True` / `harmonic_wave 1.6 ms/call` になり、Windows で使えなかった
`torch.compile` 経路が効きます。

**実測（RTX 3090 / offer $0.136 + disk 150GB で実効 $0.21 per hour、M3）:**

| 工程 | 実測 |
|---|---|
| GTSinger の取得（使う wav だけ 7,977 本 / 11 GB） | 9 分 |
| 特徴抽出 25 shard（23 話者・約 18 時間） | 52 分。**GPU 使用率 6%＝GPU 律速ではない** |
| base 学習 | 8.14 step/s、30,000 step が約 60 分 |
| peak VRAM | **1.95 GB**（24 GB 級は要らなかった） |

**実測（RTX 4090 / 実効 $0.371 per hour、M3 の継続 run）:** base 学習 **11.74 step/s**（3090 比 **1.44 倍**）、peak VRAM 2.0 GB。素材の再生成に 69.4 分、HF の取得に 23.9 分。
**実費 $2.148**（GPU $1.157 / download $0.868 / storage $0.111 / upload $0.012）。

**ディスク課金は無視できません。** 150 GB を付けたら $0.136/hr の offer が実効 $0.21/hr に
なりました（+54%）。**必要量を見積もってから付けること。** 上の構成なら実測 51 GB です。

**通信課金はもっと見落とします。** M3 の継続 run の実費 **$2.148** の内訳は
GPU $1.157 / **download $0.868** / storage $0.111 / upload $0.012 で、**通信が 40%** でした。
回線が速い offer を選びましたが、HF 側が律速で速度の恩恵は無く、単価だけ高くつきました。
**`search` の `$/TB` 列を見てから選ぶこと。**

## 4. 接続して確認する

```bash
uv run python tools/vast.py instances
uv run python tools/vast.py ssh <instance_id>
uv run python tools/vast.py logs <instance_id>
```

インスタンス上で `/root/bootstrap.log` を見る。**次を確認してから学習を始める。**

1. `is_available: True` と GPU 名・VRAM が期待どおり
2. `triton: <version>` が出ている
3. `compiled path active: True` — **False なら励起が 3〜4 倍遅いまま回ることになる**ので、
   理由（gcc が無い等）を潰してから始める
4. `unittest` が OK

不安があれば全経路を回す（3 分）:

```bash
uv run python tools/smoke/run_smoke.py
```

## 5. 学習する

[leapsinger-experiment](../leapsinger-experiment/SKILL.md) に従う。長時間になるので
バックグラウンドで回し、通知を待つ（sleep でポーリングしない）。

**M3 で初めてインスタンスを立てるときは、base 学習の前に seed 0 / seed 1 の比較を済ませる。**
手順は [`doc/svc-plan.md`](../../../doc/svc-plan.md) の M3「開始時にやること」。専用インスタンスを
立てずに済ませるための決定なので、base 学習を先に始めないこと。

## 6. 成果物を回収する（破棄の前に必ず）

**`destroy` するとディスクごと消える。取り消せない。**

回収するもの: `log/<run>/ckpt_*.pt`、`events.out.tfevents.*`、config のコピー、
`uv.lock`、`nvidia-smi` の出力、生成サンプル。

```bash
scp -i ~/.ssh/id_ed25519_vast -P <port> -r root@<host>:/root/LeapSVC/log/<run>/. ./out/
```

学習中も定期的に退避する。インスタンスは落ちることがある。

**checkpoint は学習中から回収する（今回そうして効果があった）。** 2,500 step ごとに
出るたび scp で引き取り、**学習と並行して評価まで進めました**。M4 で 40 分ぶん空回しさせた
のに対し、今回は学習完了から破棄まで数分です。**早い checkpoint で結果の傾向が見えるので、
最後まで待つ必要がない**という利点もあります（実測で 2,500 step で既に目標を超えていた）。

**回収の遅さがそのまま料金になる。** M4 では 141 MB の checkpoint 3 本の scp に時間がかかり、
**約 40 分ぶんインスタンスを空回し**させました（実費 $0.75 は見積もり $0.35 の 2.1 倍。
**超過はすべて段取りで、GPU 時間そのものは見積もりどおり**でした）。次のどちらかにします。

- **学習中から細かく回収する**（checkpoint は書かれた端から落とす）。
- **先に指標だけ回収して checkpoint を選び、必要な 1 本だけ落とす。**

大きいファイルは `sha256sum` を両側で突き合わせること。scp は**黙って途中で切れます**（実際に発生）。

**M2 で実際に回収したもの:** 生成 WAV 2 本（予測と ground-truth mel 経由）、`m2_report.json`、
TensorBoard の events、shard の `manifest.json`。checkpoint は 141 MB あるので、必要なものだけ選ぶ。

## 7. 破棄する

```bash
uv run python tools/vast.py destroy <instance_id> --yes
```

- **回収が終わったことを確認してから。** `--yes` 無しなら警告だけ出て実行されない。
- 使い終わったら必ず破棄する。止め忘れがそのまま課金になる。
- 破棄した後に `uv run python tools/vast.py instances` で残っていないことを確認する。

## やってはいけないこと

- `vastai create instance` / `vastai destroy instance` を直接叩く（料金確認と `--yes` が飛ぶ。
  hook で止まる）
- token を echo する、ログに残す、コミットする
- 回収前に破棄する
- インスタンスを立てたまま長時間の調べ物をする（手元でできることは手元でやる）
