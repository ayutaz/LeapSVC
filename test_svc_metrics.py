"""M5 ゴール 2 が要求する客観指標のうち、道具が無かった 4 つ。

**timing / CER / 信号品質 / 推論 RTF。** どれも「変換後の数値だけ」では読めないので、
このリポジトリの他の指標（`m3_verify.py` の content cos、`speaker_similarity.py` の回復率）
と同じく、**上限（GT mel をボコーダーに通した再合成）や source と並べて**読む形にします。

重いモデル（ASR / SQUIM）は**引数で受け取り**ます。単体テストはネットワークも GPU も使いません。
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np


class TimingMetricsTests(unittest.TestCase):
    """onset のずれ（M5 ゴール 2 の timing）。

    **「ずれの平均」だけでは読めません。** onset が消えた・増えた場合はペアが作れず、
    平均からは黙って抜け落ちます。**対応が付いた割合**を必ず併せて返します。
    """

    def test_identical_onsets_have_zero_deviation(self):
        from tools.timing_metrics import onset_deviation
        r = onset_deviation([0.5, 1.0, 1.5], [0.5, 1.0, 1.5], tol=0.05)
        self.assertAlmostEqual(r["median_abs_dev"], 0.0, places=9)
        self.assertEqual(r["matched"], 3)
        self.assertAlmostEqual(r["matched_ratio"], 1.0)

    def test_a_constant_shift_is_reported_as_that_shift(self):
        from tools.timing_metrics import onset_deviation
        r = onset_deviation([0.5, 1.0, 1.5], [0.52, 1.02, 1.52], tol=0.05)
        self.assertAlmostEqual(r["median_abs_dev"], 0.02, places=6)
        self.assertEqual(r["matched"], 3)

    def test_signed_bias_separates_early_from_late(self):
        # 遅れと進みが打ち消し合うと「ずれていない」に見える。符号つきも返す。
        from tools.timing_metrics import onset_deviation
        r = onset_deviation([1.0, 2.0], [1.03, 1.97], tol=0.05)
        self.assertAlmostEqual(r["median_abs_dev"], 0.03, places=6)
        self.assertAlmostEqual(r["median_signed_dev"], 0.0, places=6)

    def test_a_dropped_onset_lowers_the_matched_ratio(self):
        from tools.timing_metrics import onset_deviation
        r = onset_deviation([0.5, 1.0, 1.5], [0.5, 1.5], tol=0.05)
        self.assertEqual(r["matched"], 2)
        self.assertEqual(r["missed"], 1)
        self.assertAlmostEqual(r["matched_ratio"], 2 / 3)

    def test_an_inserted_onset_is_counted_separately(self):
        from tools.timing_metrics import onset_deviation
        r = onset_deviation([0.5, 1.0], [0.5, 0.75, 1.0], tol=0.05)
        self.assertEqual(r["spurious"], 1)
        self.assertEqual(r["matched"], 2)

    def test_an_onset_outside_the_tolerance_is_not_matched(self):
        from tools.timing_metrics import onset_deviation
        r = onset_deviation([1.0], [1.2], tol=0.05)
        self.assertEqual(r["matched"], 0)
        self.assertEqual(r["missed"], 1)
        self.assertEqual(r["spurious"], 1)

    def test_two_source_onsets_cannot_share_one_converted_onset(self):
        # **source 2 本に対し変換側が 1 本しかない場合。** 同じ onset を使い回すと
        # matched=2 になり、消えた onset（missed=1）が見えなくなる。
        from tools.timing_metrics import onset_deviation
        r = onset_deviation([1.00, 1.02], [1.01], tol=0.05)
        self.assertEqual(r["matched"], 1)
        self.assertEqual(r["missed"], 1)
        self.assertEqual(r["spurious"], 0)

    def test_a_converted_onset_is_consumed_by_the_first_match(self):
        from tools.timing_metrics import onset_deviation
        r = onset_deviation([1.00, 1.02, 1.04], [1.01, 1.03], tol=0.05)
        self.assertEqual(r["matched"], 2)
        self.assertEqual(r["missed"], 1)
        self.assertEqual(r["spurious"], 0)

    def test_matching_prefers_the_closest_candidate(self):
        from tools.timing_metrics import onset_deviation
        r = onset_deviation([1.0], [1.04, 1.001], tol=0.05)
        self.assertAlmostEqual(r["median_abs_dev"], 0.001, places=6)

    def test_no_onsets_at_all_is_reported_not_crashed(self):
        # 無声だけのクリップは実在する（chunk の 39/89 が無声だった）。0 除算で落とさない。
        from tools.timing_metrics import onset_deviation
        r = onset_deviation([], [], tol=0.05)
        self.assertEqual(r["matched"], 0)
        self.assertIsNone(r["median_abs_dev"])
        self.assertIsNone(r["matched_ratio"])

    def test_deviation_below_the_hop_resolution_cannot_be_resolved(self):
        # 実測でずれの中央値がちょうど 1 フレーム（5.8 ms）だった。分解能の下限であって
        # 「ほぼずれていない」ではない。この関係を明示しておく。
        from tools.timing_metrics import hop_resolution_seconds
        self.assertAlmostEqual(hop_resolution_seconds(44100, 256), 256 / 44100, places=9)

    def test_report_states_its_own_resolution(self):
        # 読む人が「5.8 ms のずれ」を分解能と区別できるようにする。
        from tools.timing_metrics import timing_report
        sr = 22050
        wav = np.zeros(sr, dtype=np.float32)
        wav[sr // 2: sr // 2 + 200] = np.hanning(200).astype(np.float32)
        r = timing_report(wav, wav, sr, hop=256)
        self.assertAlmostEqual(r["resolution_seconds"], 256 / sr, places=9)

    def test_detect_onsets_finds_a_pulse_train(self):
        from tools.timing_metrics import detect_onsets
        sr = 22050
        wav = np.zeros(sr * 2, dtype=np.float32)
        for t in (0.25, 0.75, 1.25, 1.75):
            i = int(t * sr)
            wav[i:i + 200] = np.hanning(200).astype(np.float32)
        got = detect_onsets(wav, sr)
        self.assertGreaterEqual(len(got), 3)
        for t in (0.25, 0.75, 1.25):
            self.assertTrue(any(abs(g - t) < 0.05 for g in got), f"{t}s の onset が出ていない")


class CerTests(unittest.TestCase):
    """明瞭度（M5 ゴール 2 の CER）。

    **歌唱の ASR は当てになりません。** したがって「変換後の CER が X%」には意味がなく、
    必ず 2 つの基準と並べます。

    | 基準 | 意味 |
    |---|---|
    | **source** | 参照テキスト。歌詞は未知なので、source の書き起こしを参照にする |
    | **上限** | GT mel をボコーダーに通した再合成。**ASR とボコーダー由来の誤りの下駄** |
    | 変換 | 変換後 |

    ASR は引数で受け取ります（単体テストで重いモデルを落とさないため）。
    """

    def test_identical_strings_have_zero_cer(self):
        from tools.asr_cer import cer
        self.assertAlmostEqual(cer("あいうえお", "あいうえお"), 0.0)

    def test_one_substitution_in_five_characters(self):
        from tools.asr_cer import cer
        self.assertAlmostEqual(cer("あいうえお", "あいうえか"), 0.2)

    def test_insertions_and_deletions_are_counted(self):
        from tools.asr_cer import cer
        self.assertAlmostEqual(cer("あいう", "あいうえ"), 1 / 3)
        self.assertAlmostEqual(cer("あいう", "あい"), 1 / 3)

    def test_cer_can_exceed_one(self):
        # 幻聴で長く書き起こすと 1 を超える。1 に丸めると「全滅」と区別できなくなる。
        from tools.asr_cer import cer
        self.assertGreater(cer("あ", "あいうえお"), 1.0)

    def test_an_empty_reference_is_rejected_rather_than_dividing_by_zero(self):
        from tools.asr_cer import cer
        with self.assertRaises(ValueError):
            cer("", "あいう")

    def test_normalization_ignores_spaces_and_punctuation(self):
        from tools.asr_cer import normalize_ja
        self.assertEqual(normalize_ja("あい、うえ お。"), "あいうえお")

    def test_normalization_folds_fullwidth_to_halfwidth(self):
        from tools.asr_cer import normalize_ja
        self.assertEqual(normalize_ja("ＡＢＣ１２３"), "abc123")

    def test_normalization_keeps_kana_distinct(self):
        # カナを潰すと別語が同じ扱いになり、CER が甘くなる。
        from tools.asr_cer import normalize_ja
        self.assertNotEqual(normalize_ja("シャツ"), normalize_ja("シヤツ"))

    def test_report_needs_the_ceiling_to_be_readable(self):
        # 上限が無いと、ASR とボコーダー由来の誤りを模型のせいにしてしまう。
        from tools.asr_cer import cer_report
        with self.assertRaises(ValueError):
            cer_report("s.wav", "c.wav", ceiling_wav=None,
                       transcribe=lambda p: "あいうえお")

    def test_report_puts_the_conversion_against_source_and_ceiling(self):
        from tools.asr_cer import cer_report
        texts = {"s.wav": "あいうえお", "c.wav": "あいうえか", "g.wav": "あいうえお"}
        r = cer_report("s.wav", "c.wav", ceiling_wav="g.wav",
                       transcribe=lambda p: texts[p])
        self.assertAlmostEqual(r["cer_converted"], 0.2)
        self.assertAlmostEqual(r["cer_ceiling"], 0.0)

    def test_report_flags_an_asr_that_produced_nothing(self):
        # 歌唱で ASR が空を返すことは実際に起こる。0% と区別できないと嘘になる。
        from tools.asr_cer import cer_report
        texts = {"s.wav": "", "c.wav": "あ", "g.wav": "あ"}
        r = cer_report("s.wav", "c.wav", ceiling_wav="g.wav", transcribe=lambda p: texts[p])
        self.assertTrue(r["asr_failed"])
        self.assertIsNone(r["cer_converted"])

    def test_an_unusable_ceiling_suppresses_the_difference(self):
        # 実測: Whisper が歌を全く書き起こせず、上限 CER も変換 CER も 1.0 になった。
        # そのとき差は 0.0 になり、「音響モデルは劣化させていない」と読めてしまう。
        # **判別できていないだけ**なので、差を出さずに旗を立てる。
        from tools.asr_cer import cer_report
        texts = {"s.wav": "あいうえお", "c.wav": "かきくけこ", "g.wav": "さしすせそ"}
        r = cer_report("s.wav", "c.wav", ceiling_wav="g.wav", transcribe=lambda p: texts[p])
        self.assertTrue(r["ceiling_unusable"])
        self.assertIsNone(r["cer_excess_over_ceiling"])

    def test_a_usable_ceiling_still_yields_the_difference(self):
        from tools.asr_cer import cer_report
        texts = {"s.wav": "あいうえおかきくけこ", "c.wav": "あいうえおかきくけさ",
                 "g.wav": "あいうえおかきくけこ"}
        r = cer_report("s.wav", "c.wav", ceiling_wav="g.wav", transcribe=lambda p: texts[p])
        self.assertFalse(r["ceiling_unusable"])
        self.assertAlmostEqual(r["cer_excess_over_ceiling"], 0.1, places=6)

    def test_the_usability_threshold_is_recorded(self):
        from tools.asr_cer import CEILING_MAX_CER, cer_report
        texts = {"s.wav": "あいうえお", "c.wav": "あいうえお", "g.wav": "あいうえお"}
        r = cer_report("s.wav", "c.wav", ceiling_wav="g.wav", transcribe=lambda p: texts[p])
        self.assertEqual(r["ceiling_max_cer"], CEILING_MAX_CER)

    def test_report_transcribes_each_clip_exactly_once(self):
        from tools.asr_cer import cer_report
        calls = []

        def tr(p):
            calls.append(p)
            return "あいうえお"
        cer_report("s.wav", "c.wav", ceiling_wav="g.wav", transcribe=tr)
        self.assertEqual(sorted(calls), ["c.wav", "g.wav", "s.wav"])

    def test_report_keeps_the_transcripts_for_inspection(self):
        # 数字だけ見て「明瞭度が落ちた」と言えない。何と聞こえたかを残す。
        from tools.asr_cer import cer_report
        texts = {"s.wav": "あいうえお", "c.wav": "あいうえか", "g.wav": "あいうえお"}
        r = cer_report("s.wav", "c.wav", ceiling_wav="g.wav", transcribe=lambda p: texts[p])
        self.assertEqual(r["transcripts"]["converted"], "あいうえか")


class SignalQualityTests(unittest.TestCase):
    """信号品質（M5 ゴール 2）。

    **歌声への妥当性は未検証です。** 使えるモデル（SQUIM / DNSMOS 系）はどれも話し声で
    学習されています。したがって**絶対値を品質として主張しません**。上限（GT mel を
    ボコーダーに通した再合成）と並べ、**そこからの差**だけを読みます。

    採点器は引数で受け取ります。
    """

    def _scorer(self, table):
        return lambda wav, sr: dict(table[float(wav[0])])

    def test_report_requires_the_ceiling(self):
        # 話し声モデルの絶対値には意味がない。上限が無いなら数値を出さない。
        from tools.signal_quality import quality_report
        wav = np.array([1.0, 0.0], dtype=np.float32)
        with self.assertRaises(ValueError):
            quality_report(wav, ceiling=None, sr=16000,
                           score=self._scorer({1.0: {"mos": 3.0}}))

    def test_report_returns_the_gap_from_the_ceiling(self):
        from tools.signal_quality import quality_report
        cnv = np.array([1.0, 0.0], dtype=np.float32)
        ceil = np.array([2.0, 0.0], dtype=np.float32)
        r = quality_report(cnv, ceiling=ceil, sr=16000,
                           score=self._scorer({1.0: {"mos": 3.2}, 2.0: {"mos": 3.8}}))
        self.assertAlmostEqual(r["converted"]["mos"], 3.2)
        self.assertAlmostEqual(r["ceiling"]["mos"], 3.8)
        self.assertAlmostEqual(r["gap"]["mos"], -0.6, places=6)

    def test_every_shared_metric_gets_a_gap(self):
        from tools.signal_quality import quality_report
        cnv = np.array([1.0, 0.0], dtype=np.float32)
        ceil = np.array([2.0, 0.0], dtype=np.float32)
        r = quality_report(cnv, ceiling=ceil, sr=16000, score=self._scorer(
            {1.0: {"mos": 3.0, "stoi": 0.80}, 2.0: {"mos": 3.5, "stoi": 0.90}}))
        self.assertEqual(sorted(r["gap"]), ["mos", "stoi"])

    def test_a_metric_missing_from_one_side_is_not_silently_dropped(self):
        # 片側にしか無い指標の差を 0 と書くと、測れていないことが見えなくなる。
        from tools.signal_quality import quality_report
        cnv = np.array([1.0, 0.0], dtype=np.float32)
        ceil = np.array([2.0, 0.0], dtype=np.float32)
        r = quality_report(cnv, ceiling=ceil, sr=16000, score=self._scorer(
            {1.0: {"mos": 3.0, "stoi": 0.8}, 2.0: {"mos": 3.5}}))
        self.assertNotIn("stoi", r["gap"])
        self.assertIn("stoi", r["unpaired"])

    def test_report_carries_the_domain_caveat(self):
        # この数値を読む人が、話し声モデルだと知らずに引用しないようにする。
        from tools.signal_quality import quality_report
        cnv = np.array([1.0, 0.0], dtype=np.float32)
        ceil = np.array([2.0, 0.0], dtype=np.float32)
        r = quality_report(cnv, ceiling=ceil, sr=16000,
                           score=self._scorer({1.0: {"mos": 3.0}, 2.0: {"mos": 3.5}}))
        self.assertIn("caveat", r)
        self.assertIn("話し声", r["caveat"])

    def test_each_clip_is_scored_exactly_once(self):
        from tools.signal_quality import quality_report
        cnv = np.array([1.0, 0.0], dtype=np.float32)
        ceil = np.array([2.0, 0.0], dtype=np.float32)
        calls = []

        def score(wav, sr):
            calls.append(float(wav[0]))
            return {"mos": 3.0}
        quality_report(cnv, ceiling=ceil, sr=16000, score=score)
        self.assertEqual(sorted(calls), [1.0, 2.0])


class RtfTests(unittest.TestCase):
    """推論の RTF と peak VRAM（M5 ゴール 2）。

    **ゴール 2 は「特徴抽出と vocoder を含むか除くかを併記」を要求します。** 単一の RTF を
    出すと、どこまでを数えたかで 2 倍以上変わるため比較できません。**段ごとの内訳**を持たせ、
    含む / 除くの両方を計算します。

    段の実行そのものは引数で受け取ります（重いモデルを単体テストで動かさないため）。
    """

    def test_rtf_is_compute_time_over_audio_duration(self):
        from tools.rtf import rtf_from_stages
        r = rtf_from_stages({"flow": 1.0}, audio_seconds=2.0)
        self.assertAlmostEqual(r["rtf_total"], 0.5)

    def test_stages_are_summed(self):
        from tools.rtf import rtf_from_stages
        r = rtf_from_stages({"content": 0.5, "f0": 0.25, "flow": 0.25}, audio_seconds=2.0)
        self.assertAlmostEqual(r["rtf_total"], 0.5)

    def test_acoustic_only_excludes_features_and_vocoder(self):
        # 「1-step だから速い」を示すには acoustic だけの数字が要る。全体と混同しない。
        from tools.rtf import rtf_from_stages
        r = rtf_from_stages({"content": 1.0, "f0": 1.0, "flow": 0.5, "vocoder": 1.5},
                            audio_seconds=2.0)
        self.assertAlmostEqual(r["rtf_acoustic_only"], 0.25)
        self.assertAlmostEqual(r["rtf_total"], 2.0)

    def test_the_breakdown_keeps_every_stage(self):
        from tools.rtf import rtf_from_stages
        r = rtf_from_stages({"content": 1.0, "flow": 1.0}, audio_seconds=1.0)
        self.assertEqual(sorted(r["stage_seconds"]), ["content", "flow"])
        self.assertAlmostEqual(r["stage_rtf"]["content"], 1.0)

    def test_a_stage_that_is_not_a_known_category_is_rejected(self):
        # 未知の段を黙って「その他」に入れると、含む/除くの境界が曖昧になる。
        from tools.rtf import rtf_from_stages
        with self.assertRaises(ValueError):
            rtf_from_stages({"mystery": 1.0}, audio_seconds=1.0)

    def test_zero_length_audio_is_rejected(self):
        from tools.rtf import rtf_from_stages
        with self.assertRaises(ValueError):
            rtf_from_stages({"flow": 1.0}, audio_seconds=0.0)

    def test_realtime_claim_needs_end_to_end_not_acoustic_only(self):
        # doc の主張ルール: 「リアルタイム」は end-to-end 実測の後だけ。
        # acoustic だけが 1 を切っても total が超えていれば realtime_capable は False。
        from tools.rtf import rtf_from_stages
        r = rtf_from_stages({"content": 1.0, "flow": 0.1, "vocoder": 0.5}, audio_seconds=1.0)
        self.assertFalse(r["realtime_capable"])
        self.assertLess(r["rtf_acoustic_only"], 1.0)

    def test_measure_stage_records_wall_time_and_result(self):
        from tools.rtf import measure_stage
        out, sec = measure_stage(lambda: 42)
        self.assertEqual(out, 42)
        self.assertGreaterEqual(sec, 0.0)

    def test_repeats_take_the_median_not_the_first_run(self):
        # 初回は重みの読み込みや JIT を含む。それを RTF として報告しない。
        from tools.rtf import measure_stage
        calls = []

        def f():
            calls.append(1)
            return len(calls)
        out, sec = measure_stage(f, repeats=3)
        self.assertEqual(len(calls), 3)
        self.assertEqual(out, 3)


class TestSetTests(unittest.TestCase):
    """M5 の test set を決定的に選ぶ（[実行計画](doc/svc-plan.md) M5「決定 4」）。

    **手で選ぶと偏ります。** M3 の測り直しで、`rglob` の並び順のまま 20 clip を取ったら
    **18 女性 / 2 男性**になりました。事前登録した構成（同性・異性の両方、日本語を含む、
    12 秒以上、held-out song と未知 source の両方）を**コードで満たさせます**。

    選択は seed から決定的に決まり、manifest に残せること。
    """

    def _pool(self, n_female=9, n_male=11, per_speaker=3):
        return [{"speaker": f"{g}{i}", "gender": g, "clip": f"c{k}", "seconds": 20.0,
                 "kind": "unseen"}
                for g, n in (("female", n_female), ("male", n_male))
                for i in range(1, n + 1) for k in range(per_speaker)]

    def test_every_speaker_contributes_exactly_one_clip(self):
        # VocalSet は女性 9 / 男性 11。全 20 名から 1 本ずつで n=20 になる。
        from tools.m5_testset import select_clips
        got = select_clips(self._pool(), n=20, seed=0)
        self.assertEqual(len({c["speaker"] for c in got}), 20)

    def test_both_genders_clear_the_floor(self):
        # 性別で回復率の伸びが倍違うので、片方が少なすぎると差を読めない。
        from tools.m5_testset import select_clips
        got = select_clips(self._pool(), n=20, seed=0)
        genders = [c["gender"] for c in got]
        for g in ("female", "male"):
            self.assertGreaterEqual(genders.count(g), 6, f"{g} が少なすぎる")

    def test_selection_is_deterministic_for_a_seed(self):
        from tools.m5_testset import select_clips
        pool = self._pool()
        a = [c["speaker"] + c["clip"] for c in select_clips(pool, n=20, seed=0)]
        b = [c["speaker"] + c["clip"] for c in select_clips(pool, n=20, seed=0)]
        self.assertEqual(a, b)

    def test_a_different_seed_gives_a_different_set(self):
        from tools.m5_testset import select_clips
        pool = self._pool()
        a = {c["speaker"] + c["clip"] for c in select_clips(pool, n=20, seed=0)}
        b = {c["speaker"] + c["clip"] for c in select_clips(pool, n=20, seed=1)}
        self.assertNotEqual(a, b)

    def test_clips_shorter_than_the_calibrated_length_are_dropped(self):
        # 話者類似度の較正は 12 秒以上でしか通らない。短い clip を入れると測れなくなる。
        from tools.m5_testset import select_clips
        pool = self._pool()
        for c in pool:
            c["seconds"] = 6.0
        with self.assertRaises(ValueError):
            select_clips(pool, n=20, seed=0)

    def test_no_speaker_appears_twice(self):
        # 1 人から複数取ると、その歌手の癖が test set を支配する。
        from tools.m5_testset import select_clips
        got = select_clips(self._pool(), n=20, seed=0)
        counts = {}
        for c in got:
            counts[c["speaker"]] = counts.get(c["speaker"], 0) + 1
        self.assertLessEqual(max(counts.values()), 1)

    def test_it_refuses_when_there_are_not_enough_speakers(self):
        # 足りないなら黙って 1 人から 2 本取らず、止める。
        from tools.m5_testset import select_clips
        with self.assertRaises(ValueError):
            select_clips(self._pool(n_female=5, n_male=5, per_speaker=4), n=20, seed=0)

    def test_it_refuses_a_set_that_is_lopsided_by_gender(self):
        # M3 では 18 女性 / 2 男性の set で「男性 source が暗い」を取り逃がした。
        # 素材が偏っているなら、黙って偏った set を返さずに止める。
        from tools.m5_testset import select_clips
        with self.assertRaises(ValueError):
            select_clips(self._pool(n_female=18, n_male=2, per_speaker=1), n=20, seed=0)

    def test_target_holdout_clips_are_kept_separate_from_unseen(self):
        # ゴール 1 は held-out song と未知 source singer の**両方**を要求する。
        from tools.m5_testset import build_testset
        unseen = self._pool()
        holdout = [{"speaker": "ritsu", "gender": "female", "clip": f"h{k}",
                    "seconds": 20.0, "kind": "holdout"} for k in range(8)]
        ts = build_testset(unseen, holdout, n_unseen=20, n_holdout=6, seed=0)
        self.assertEqual(len(ts["unseen"]), 20)
        self.assertEqual(len(ts["holdout"]), 6)
        self.assertTrue(all(c["kind"] == "holdout" for c in ts["holdout"]))

    def test_holdout_takes_several_segments_from_one_long_song(self):
        # hold-out 曲は 3 曲しかないが 1 曲 3〜5 分ある。曲単位 split を保ったまま
        # 区間を分けて 6 clip にする。**別の曲を混ぜて水増ししない。**
        from tools.m5_testset import segment_holdout
        songs = [{"speaker": "ritsu", "gender": "female", "song": "a", "path": "a.wav",
                  "full_seconds": 280.0, "kind": "holdout"},
                 {"speaker": "ritsu", "gender": "female", "song": "b", "path": "b.wav",
                  "full_seconds": 200.0, "kind": "holdout"}]
        got = segment_holdout(songs, n=4, seconds=20.0, seed=0)
        self.assertEqual(len(got), 4)
        self.assertEqual(sorted({c["song"] for c in got}), ["a", "b"])

    def test_segments_from_the_same_song_do_not_overlap(self):
        # 同じ区間を 2 回測ると n を水増ししただけになる。
        from tools.m5_testset import segment_holdout
        songs = [{"speaker": "ritsu", "gender": "female", "song": "a", "path": "a.wav",
                  "full_seconds": 100.0, "kind": "holdout"}]
        got = segment_holdout(songs, n=4, seconds=20.0, seed=0)
        spans = sorted((c["start"], c["start"] + c["seconds"]) for c in got)
        for (_s1, e1), (s2, _s2) in zip(spans, spans[1:], strict=False):
            self.assertLessEqual(e1, s2, f"区間が重なっている: {spans}")

    def test_segments_are_spread_over_each_song(self):
        # 曲頭に固まるとイントロばかりになる（無声率が高く、変換の検証に向かない）。
        from tools.m5_testset import segment_holdout
        songs = [{"speaker": "ritsu", "gender": "female", "song": "a", "path": "a.wav",
                  "full_seconds": 300.0, "kind": "holdout"}]
        got = segment_holdout(songs, n=3, seconds=20.0, seed=0)
        self.assertGreater(max(c["start"] for c in got), 100.0)

    def test_segments_are_chosen_where_the_singer_is_actually_singing(self):
        # **実際に踏んだ。** 曲を等分して窓の中でずらすと、イントロや間奏が選ばれる。
        # hold-out 6 本のうち 3 本が有声率 3.5〜34% になり、話者性を測れなくなった
        # （リツ本人の source が参照と 0.39 しか一致しない = 上限 0.67 から大きく外れる）。
        from tools.m5_testset import segment_holdout
        # 前半 100 秒は無声、後半 200 秒が有声、という曲を模す
        def voiced_ratio(path, start, seconds):
            return 0.05 if start < 100.0 else 0.9
        songs = [{"speaker": "ritsu", "gender": "female", "song": "a", "path": "a.wav",
                  "full_seconds": 300.0, "kind": "holdout"}]
        got = segment_holdout(songs, n=3, seconds=20.0, seed=0,
                              voiced_ratio=voiced_ratio, min_voiced=0.5)
        for c in got:
            self.assertGreaterEqual(c["start"], 100.0, f"無声区間が選ばれた: {c}")

    def test_the_voiced_ratio_is_recorded_for_each_segment(self):
        # 後から「なぜこの区間か」を見られるようにする。
        from tools.m5_testset import segment_holdout
        songs = [{"speaker": "ritsu", "gender": "female", "song": "a", "path": "a.wav",
                  "full_seconds": 300.0, "kind": "holdout"}]
        got = segment_holdout(songs, n=2, seconds=20.0, seed=0,
                              voiced_ratio=lambda p, s, d: 0.8, min_voiced=0.5)
        self.assertAlmostEqual(got[0]["voiced_ratio"], 0.8)

    def test_it_refuses_when_no_segment_is_voiced_enough(self):
        # 黙って無声区間を返さない。素材か閾値を見直させる。
        from tools.m5_testset import segment_holdout
        songs = [{"speaker": "ritsu", "gender": "female", "song": "a", "path": "a.wav",
                  "full_seconds": 300.0, "kind": "holdout"}]
        with self.assertRaises(ValueError):
            segment_holdout(songs, n=2, seconds=20.0, seed=0,
                            voiced_ratio=lambda p, s, d: 0.1, min_voiced=0.5)

    def test_without_a_voiced_check_the_old_behaviour_is_kept(self):
        from tools.m5_testset import segment_holdout
        songs = [{"speaker": "ritsu", "gender": "female", "song": "a", "path": "a.wav",
                  "full_seconds": 300.0, "kind": "holdout"}]
        got = segment_holdout(songs, n=3, seconds=20.0, seed=0)
        self.assertEqual(len(got), 3)

    def test_it_refuses_a_song_too_short_to_yield_its_share(self):
        from tools.m5_testset import segment_holdout
        songs = [{"speaker": "ritsu", "gender": "female", "song": "a", "path": "a.wav",
                  "full_seconds": 25.0, "kind": "holdout"}]
        with self.assertRaises(ValueError):
            segment_holdout(songs, n=4, seconds=20.0, seed=0)

    def test_the_manifest_records_the_seed_and_the_rule(self):
        # 後から「なぜこの 20 本か」を復元できないと、比較の土台が消える。
        from tools.m5_testset import build_testset
        unseen = self._pool()
        holdout = [{"speaker": "ritsu", "gender": "female", "clip": f"h{k}",
                    "seconds": 20.0, "kind": "holdout"} for k in range(8)]
        ts = build_testset(unseen, holdout, n_unseen=20, n_holdout=6, seed=3)
        self.assertEqual(ts["manifest"]["seed"], 3)
        self.assertEqual(ts["manifest"]["min_seconds"], 12.0)
        self.assertEqual(ts["manifest"]["n_unseen"], 20)

    def test_transpose_is_assigned_by_gender_not_left_to_the_operator(self):
        # 男性 source を移調し忘れると、モデルではなく音域差を測ることになる。
        from tools.m5_testset import MALE_TRANSPOSE, build_testset
        unseen = self._pool()
        holdout = [{"speaker": "ritsu", "gender": "female", "clip": f"h{k}",
                    "seconds": 20.0, "kind": "holdout"} for k in range(8)]
        ts = build_testset(unseen, holdout, n_unseen=20, n_holdout=6, seed=0)
        for c in ts["unseen"]:
            # **値をハードコードしない。** 掃引で変わる（2026-09-16 に +12 -> +7）。
            self.assertEqual(c["transpose"],
                             MALE_TRANSPOSE if c["gender"] == "male" else 0)

    def test_holdout_clips_are_never_transposed(self):
        # target 自身の曲は音域が合っている。移調すると別の実験になる。
        from tools.m5_testset import build_testset
        unseen = self._pool()
        holdout = [{"speaker": "ritsu", "gender": "female", "clip": f"h{k}",
                    "seconds": 20.0, "kind": "holdout"} for k in range(8)]
        ts = build_testset(unseen, holdout, n_unseen=20, n_holdout=6, seed=0)
        self.assertTrue(all(c["transpose"] == 0 for c in ts["holdout"]))


class ConvertOutputNamingTests(unittest.TestCase):
    """変換結果のファイル名（M5 で同じ曲から複数区間を取るため）。

    **同じ WAV から区間を変えて 2 回変換すると、出力名が衝突します。** hold-out は 3 曲しか
    なく、1 曲から 2 区間取るのでこれが実際に起きます。**黙って上書きされると clip 数が
    減るだけで例外にならない**（GTSinger で 1,922 ファイルが 3 名に潰れたのと同じ形）。
    """

    def test_the_stem_defaults_to_the_input_filename(self):
        from tools.svc_convert import output_stem
        self.assertEqual(output_stem("download/ritsu/anywhere.wav", tag=None), "anywhere")

    def test_a_tag_disambiguates_two_segments_of_one_song(self):
        from tools.svc_convert import output_stem
        a = output_stem("a/anywhere.wav", tag="seg0")
        b = output_stem("a/anywhere.wav", tag="seg1")
        self.assertNotEqual(a, b)

    def test_the_tag_keeps_the_source_name_readable(self):
        # 出力を見て、どの曲のどの区間かが分かること。
        from tools.svc_convert import output_stem
        self.assertIn("anywhere", output_stem("a/anywhere.wav", tag="seg1"))

    def test_the_tag_is_sanitised_for_the_filesystem(self):
        from tools.svc_convert import output_stem
        got = output_stem("a/anywhere.wav", tag="seg/1 2")
        for bad in r"/\ ":
            self.assertNotIn(bad, got)


class BlindPreferenceTests(unittest.TestCase):
    """blind preference test の道具（M5 ゴール 3）。

    **N=1 でも blind の条件は要ります。** ラベルを隠し、順序を randomize し、
    どちらが A でどちらが B だったかを**後から復元できる**こと。復元できないと
    集計そのものが成り立ちません。

    **聴く前に答えが分かってはいけない**ので、割り当ては seed から決まりつつ、
    提示側のファイル名からは system が読めない形にします。
    """

    def _pairs(self, n=4):
        return [{"clip": f"c{i}", "a_system": "leapsvc", "b_system": "seedvc"}
                for i in range(n)]

    def test_each_pair_gets_a_random_side_assignment(self):
        from tools.blind_test import assign_sides
        got = assign_sides(["c0", "c1", "c2", "c3"], systems=("leapsvc", "seedvc"), seed=0)
        self.assertEqual(len(got), 4)
        for row in got:
            self.assertEqual(sorted([row["A"], row["B"]]), ["leapsvc", "seedvc"])

    def test_the_assignment_is_deterministic_for_a_seed(self):
        from tools.blind_test import assign_sides
        a = assign_sides(["c0", "c1"], systems=("x", "y"), seed=7)
        b = assign_sides(["c0", "c1"], systems=("x", "y"), seed=7)
        self.assertEqual(a, b)

    def test_both_systems_appear_on_each_side_across_the_set(self):
        # 常に A が LeapSVC だと、順序の癖が preference に化ける。
        from tools.blind_test import assign_sides
        got = assign_sides([f"c{i}" for i in range(20)], systems=("x", "y"), seed=0)
        a_sides = [r["A"] for r in got]
        self.assertGreater(a_sides.count("x"), 3)
        self.assertGreater(a_sides.count("y"), 3)

    def test_presentation_order_is_shuffled_not_the_clip_order(self):
        # clip の並びも混ぜる。曲順で聴くと後半に慣れが出る。
        from tools.blind_test import assign_sides
        clips = [f"c{i}" for i in range(20)]
        got = [r["clip"] for r in assign_sides(clips, systems=("x", "y"), seed=1)]
        self.assertNotEqual(got, clips)
        self.assertEqual(sorted(got), sorted(clips))

    def test_clip_tags_are_read_from_either_naming(self):
        # LeapSVC は `<name>__<tag>_converted.wav`、正規化した baseline は
        # `<tag>_converted.wav`。**どちらからも同じ tag が取れること。**
        from tools.blind_test import clip_tag
        self.assertEqual(clip_tag("m10_dona__unseen00_converted.wav"), "unseen00")
        self.assertEqual(clip_tag("unseen00_converted.wav"), "unseen00")

    def test_finding_clips_pairs_the_two_systems_by_tag(self):
        from tools.blind_test import find_clips
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "a").mkdir()
            (root / "b").mkdir()
            (root / "a" / "song__unseen00_converted.wav").write_bytes(b"x")
            (root / "b" / "unseen00_converted.wav").write_bytes(b"x")
            a, b = find_clips(root / "a"), find_clips(root / "b")
            self.assertEqual(sorted(set(a) & set(b)), ["unseen00"])

    def test_source_and_ceiling_files_are_not_treated_as_clips(self):
        from tools.blind_test import find_clips
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for n in ("x__unseen00_converted.wav", "x__unseen00_source.wav",
                      "x__unseen00_vocoder_only.wav"):
                (root / n).write_bytes(b"x")
            self.assertEqual(sorted(find_clips(root)), ["unseen00"])

    def test_loudness_is_matched_before_presenting(self):
        """**実際に踏んだ。** LeapSVC の出力は Seed-VC の 5.4 倍大きく（RMS 0.143 対 0.027）、
        **音量だけで system を当てられる状態**でした。事前登録した条件は「両システムの出力に
        同じ loudness 揃えを当てる」です。人は大きいほうを好むので、揃えないと preference が
        音量の選好になります。
        """
        from tools.blind_test import match_loudness
        a = (np.ones(1000, dtype=np.float32) * 0.5)
        b = (np.ones(1000, dtype=np.float32) * 0.05)
        ma, mb = match_loudness(a, b)

        def rms(x):
            return float(np.sqrt(np.mean(x ** 2)))
        self.assertAlmostEqual(rms(ma), rms(mb), places=4)

    def test_matching_does_not_clip(self):
        # 揃えるために持ち上げて 1.0 を超えると、歪みが preference に化ける。
        from tools.blind_test import match_loudness
        a = np.ones(1000, dtype=np.float32) * 0.9
        b = np.ones(1000, dtype=np.float32) * 0.05
        for x in match_loudness(a, b):
            self.assertLessEqual(float(np.abs(x).max()), 1.0)

    def test_the_quieter_side_is_raised_not_only_the_louder_lowered(self):
        # 両方下げると聴きづらくなる。**目標 RMS へ揃える**。
        from tools.blind_test import match_loudness
        a = np.ones(1000, dtype=np.float32) * 0.5
        b = np.ones(1000, dtype=np.float32) * 0.05
        ma, mb = match_loudness(a, b)
        self.assertGreater(float(np.abs(mb).max()), 0.05)

    def test_silence_does_not_blow_up(self):
        from tools.blind_test import match_loudness
        a = np.zeros(1000, dtype=np.float32)
        b = np.ones(1000, dtype=np.float32) * 0.1
        for x in match_loudness(a, b):
            self.assertTrue(np.all(np.isfinite(x)))

    def test_a_paired_file_puts_a_then_b_with_a_gap(self):
        """26 ペアを A/B 別ファイルで聴くと切り替えの手間が大きい。**1 本に繋いだ形**も
        用意します。**A -> 無音 -> B** の順で、境目が分かるようにします。
        """
        from tools.blind_test import concat_pair
        a = np.ones(100, dtype=np.float32) * 0.5
        b = np.ones(100, dtype=np.float32) * 0.3
        out = concat_pair(a, b, sr=100, gap_sec=0.5)
        self.assertEqual(len(out), 100 + 50 + 100)
        self.assertAlmostEqual(float(out[0]), 0.5, places=5)
        self.assertAlmostEqual(float(out[-1]), 0.3, places=5)

    def test_the_gap_is_actually_silent(self):
        from tools.blind_test import concat_pair
        a = np.ones(100, dtype=np.float32) * 0.5
        b = np.ones(100, dtype=np.float32) * 0.3
        out = concat_pair(a, b, sr=100, gap_sec=0.5)
        self.assertEqual(float(np.abs(out[100:150]).max()), 0.0)

    def test_concatenation_does_not_renormalise(self):
        # ここで音量を触ると、揃えた意味が消える。
        from tools.blind_test import concat_pair
        a = np.ones(100, dtype=np.float32) * 0.5
        b = np.ones(100, dtype=np.float32) * 0.3
        out = concat_pair(a, b, sr=100, gap_sec=0.1)
        self.assertAlmostEqual(float(out[:100].max()), 0.5, places=5)
        self.assertAlmostEqual(float(out[-100:].max()), 0.3, places=5)

    def test_tally_counts_votes_per_system_not_per_side(self):
        from tools.blind_test import tally
        sheet = [{"clip": "c0", "A": "x", "B": "y", "vote": "A"},
                 {"clip": "c1", "A": "y", "B": "x", "vote": "A"},
                 {"clip": "c2", "A": "x", "B": "y", "vote": "B"}]
        got = tally(sheet)
        self.assertEqual(got["wins"]["x"], 1)
        self.assertEqual(got["wins"]["y"], 2)

    def test_ties_are_kept_not_dropped(self):
        # 引き分けを捨てると、差が無かったことが見えなくなる。
        from tools.blind_test import tally
        sheet = [{"clip": "c0", "A": "x", "B": "y", "vote": "tie"},
                 {"clip": "c1", "A": "x", "B": "y", "vote": "A"}]
        got = tally(sheet)
        self.assertEqual(got["ties"], 1)
        self.assertEqual(got["n_voted"], 2)

    def test_an_unvoted_row_is_reported_not_silently_skipped(self):
        from tools.blind_test import tally
        sheet = [{"clip": "c0", "A": "x", "B": "y", "vote": ""},
                 {"clip": "c1", "A": "x", "B": "y", "vote": "A"}]
        got = tally(sheet)
        self.assertEqual(got["n_missing"], 1)
        self.assertEqual(got["n_voted"], 1)

    def test_an_invalid_vote_is_rejected_rather_than_guessed(self):
        from tools.blind_test import tally
        with self.assertRaises(ValueError):
            tally([{"clip": "c0", "A": "x", "B": "y", "vote": "leapsvc"}])

    def test_the_sign_test_p_value_is_reported_with_n(self):
        # N=1 の評価者なので、統計は「参考」。それでも n と p を出して読み手に判断させる。
        from tools.blind_test import tally
        sheet = [{"clip": f"c{i}", "A": "x", "B": "y", "vote": "A"} for i in range(10)]
        got = tally(sheet)
        self.assertEqual(got["n_decisive"], 10)
        self.assertLess(got["p_two_sided"], 0.01)

    def test_a_split_result_is_not_significant(self):
        from tools.blind_test import tally
        sheet = ([{"clip": f"a{i}", "A": "x", "B": "y", "vote": "A"} for i in range(5)]
                 + [{"clip": f"b{i}", "A": "x", "B": "y", "vote": "B"} for i in range(5)])
        got = tally(sheet)
        self.assertGreater(got["p_two_sided"], 0.5)


class NormaliseOutputsTests(unittest.TestCase):
    """外部 baseline の出力を、測定ツールが読める形へ揃える（M5）。

    Seed-VC は `<tag>/vc_<tag>__<ref>_<...>.wav` に書き、LeapSVC は
    `<name>__<tag>_converted.wav` に書きます。**測定ツールは後者の形しか読みません。**

    **source をコピーで作らず、必ず同じ区間から取ること。** timing と CER は source を
    基準にするので、別の区間を source として置くと比較が壊れます。
    """

    def test_it_pairs_each_tag_with_its_converted_file(self):
        from tools.normalise_outputs import find_seedvc_outputs
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for tag in ("unseen00", "holdout01"):
                (root / tag).mkdir()
                (root / tag / f"vc_{tag}__ref_1.0_30_0.7.wav").write_bytes(b"x")
            got = find_seedvc_outputs(root)
            self.assertEqual(sorted(got), ["holdout01", "unseen00"])

    def test_it_ignores_the_segments_and_reference_files(self):
        from tools.normalise_outputs import find_seedvc_outputs
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "_segments").mkdir()
            (root / "_segments" / "unseen00.wav").write_bytes(b"x")
            (root / "_target_ref.wav").write_bytes(b"x")
            (root / "unseen01").mkdir()
            (root / "unseen01" / "vc_unseen01__ref.wav").write_bytes(b"x")
            self.assertEqual(sorted(find_seedvc_outputs(root)), ["unseen01"])

    def test_a_tag_with_two_outputs_is_rejected_rather_than_guessed(self):
        # 設定違いの出力が両方残っていると、どちらを測ったのか分からなくなる。
        from tools.normalise_outputs import find_seedvc_outputs
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "unseen00").mkdir()
            (root / "unseen00" / "vc_unseen00__ref_1.0_30_0.7.wav").write_bytes(b"x")
            (root / "unseen00" / "vc_unseen00__ref_1.0_50_0.7.wav").write_bytes(b"x")
            with self.assertRaises(ValueError):
                find_seedvc_outputs(root)

    def test_the_normalised_name_matches_what_the_metrics_expect(self):
        from tools.normalise_outputs import normalised_name
        self.assertEqual(normalised_name("unseen00", "converted"), "unseen00_converted.wav")
        self.assertEqual(normalised_name("unseen00", "source"), "unseen00_source.wav")


class GuardRailTests(unittest.TestCase):
    """事前登録した guard rail 判定（[実行計画](doc/svc-plan.md) M5「決定 1」）。

    **絶対閾値は置きません。** 「悪化」= **clip 間のばらつき（標準誤差）を超えて相手より低い**。
    手元のデータ量では任意の数字を発明することになるためです。

    **preference で勝っても guard rail を 1 つでも落としていたら「より良い」とは書きません。**
    その判断をコードに書いて、後から緩められないようにします。
    """

    def test_a_clear_drop_beyond_the_standard_error_is_flagged(self):
        from tools.guard_rail import compare_metric
        ours = [0.50] * 10
        theirs = [0.80] * 10
        r = compare_metric(ours, theirs, higher_is_better=True)
        self.assertTrue(r["worse"])

    def test_a_difference_inside_the_noise_is_not_flagged(self):
        from tools.guard_rail import compare_metric
        rng = np.random.default_rng(0)
        ours = list(rng.normal(0.80, 0.10, 20))
        theirs = list(rng.normal(0.81, 0.10, 20))
        r = compare_metric(ours, theirs, higher_is_better=True)
        self.assertFalse(r["worse"])

    def test_being_better_is_never_flagged_as_worse(self):
        from tools.guard_rail import compare_metric
        r = compare_metric([0.9] * 10, [0.5] * 10, higher_is_better=True)
        self.assertFalse(r["worse"])

    def test_lower_is_better_metrics_flip_the_direction(self):
        # CER やずれは小さいほうが良い。方向を取り違えると結論が反転する。
        from tools.guard_rail import compare_metric
        r = compare_metric([0.9] * 10, [0.2] * 10, higher_is_better=False)
        self.assertTrue(r["worse"])

    def test_the_margin_and_threshold_are_reported(self):
        from tools.guard_rail import compare_metric
        r = compare_metric([0.5] * 10, [0.8] * 10, higher_is_better=True)
        self.assertIn("diff", r)
        self.assertIn("threshold", r)
        self.assertLess(r["diff"], 0)

    def test_a_verdict_requires_every_rail_to_hold(self):
        from tools.guard_rail import verdict
        rails = {"content": {"worse": False}, "similarity": {"worse": True}}
        v = verdict(rails, preference_winner="ours", ours="ours")
        self.assertFalse(v["may_claim_better"])
        self.assertEqual(v["failed_rails"], ["similarity"])

    def test_winning_preference_with_all_rails_intact_allows_the_claim(self):
        from tools.guard_rail import verdict
        rails = {"content": {"worse": False}, "similarity": {"worse": False}}
        v = verdict(rails, preference_winner="ours", ours="ours")
        self.assertTrue(v["may_claim_better"])

    def test_losing_preference_never_allows_the_claim(self):
        from tools.guard_rail import verdict
        rails = {"content": {"worse": False}}
        v = verdict(rails, preference_winner="theirs", ours="ours")
        self.assertFalse(v["may_claim_better"])

    def test_a_tie_in_preference_does_not_allow_the_claim(self):
        from tools.guard_rail import verdict
        rails = {"content": {"worse": False}}
        v = verdict(rails, preference_winner=None, ours="ours")
        self.assertFalse(v["may_claim_better"])


class BatchConvertTests(unittest.TestCase):
    """複数 clip を 1 プロセスで変換する（M5 の 26 clip を現実的な時間で回すため）。

    **律速はモデルの読み込みです。** 実測で GPU 使用率は 5% しかなく、1 clip ごとに
    ContentVec と RMVPE を読み直すぶんが所要のほとんどでした。**モデルは 1 度だけ読み、
    全 clip に使い回します。**

    重いモデルは引数で受け取るので、このテストはネットワークも GPU も使いません。
    """

    def _job(self, tag, tp=0, start=0.0, seconds=1.0):
        return {"tag": tag, "path": f"{tag}.wav", "transpose": tp,
                "start": start, "seconds": seconds}

    def test_models_are_loaded_once_for_the_whole_batch(self):
        from tools.svc_batch import run_batch
        loads = []

        def load_models():
            loads.append(1)
            return {"model": "m"}
        run_batch([self._job("a"), self._job("b"), self._job("c")],
                  load_models=load_models, convert_one=lambda job, models: None)
        self.assertEqual(len(loads), 1)

    def test_each_job_is_converted_once(self):
        from tools.svc_batch import run_batch
        done = []
        run_batch([self._job("a"), self._job("b")],
                  load_models=lambda: {}, convert_one=lambda job, m: done.append(job["tag"]))
        self.assertEqual(done, ["a", "b"])

    def test_a_failing_clip_does_not_abort_the_rest(self):
        # 26 本の途中で 1 本落ちたときに、そこまでの結果を捨てない。
        from tools.svc_batch import run_batch
        done = []

        def convert(job, m):
            if job["tag"] == "b":
                raise RuntimeError("boom")
            done.append(job["tag"])
        rep = run_batch([self._job("a"), self._job("b"), self._job("c")],
                        load_models=lambda: {}, convert_one=convert)
        self.assertEqual(done, ["a", "c"])
        self.assertEqual(rep["failed"], ["b"])

    def test_the_failure_reason_is_kept(self):
        from tools.svc_batch import run_batch

        def convert(job, m):
            raise RuntimeError("特定の理由")
        rep = run_batch([self._job("a")], load_models=lambda: {}, convert_one=convert)
        self.assertIn("特定の理由", rep["errors"]["a"])

    def test_already_converted_clips_are_skipped(self):
        # 途中から再開できること。26 本を最初からやり直さない。
        from tools.svc_batch import run_batch
        done = []
        rep = run_batch([self._job("a"), self._job("b")],
                        load_models=lambda: {},
                        convert_one=lambda job, m: done.append(job["tag"]),
                        is_done=lambda job: job["tag"] == "a")
        self.assertEqual(done, ["b"])
        self.assertEqual(rep["skipped"], ["a"])

    def test_models_are_not_loaded_when_everything_is_already_done(self):
        # 全部済んでいるのに数百 MB を読み込まない。
        from tools.svc_batch import run_batch
        loads = []
        run_batch([self._job("a")], load_models=lambda: loads.append(1),
                  convert_one=lambda job, m: None, is_done=lambda job: True)
        self.assertEqual(loads, [])

    def test_a_float_transpose_does_not_break_the_progress_line(self):
        # **実際に踏んだ。** jobs.tsv は transpose を float で持つのに、進捗表示が :+3d
        # だったので、**書き出しの後**に例外が出て 16 本が「失敗」と記録された。
        # 出力は完全なのに失敗と報告されるのが最も危ない（再実行しても is_done で飛ばされる）。
        from tools.svc_batch import format_progress
        line = format_progress({"tag": "unseen19", "transpose": 12.0}, seconds=20.0,
                               elapsed=41.2)
        self.assertIn("unseen19", line)
        self.assertIn("+12", line)

    def test_an_int_transpose_formats_the_same_way(self):
        from tools.svc_batch import format_progress
        a = format_progress({"tag": "t", "transpose": 12}, seconds=1.0, elapsed=1.0)
        b = format_progress({"tag": "t", "transpose": 12.0}, seconds=1.0, elapsed=1.0)
        self.assertEqual(a, b)

    def test_a_negative_transpose_keeps_its_sign(self):
        from tools.svc_batch import format_progress
        self.assertIn("-7", format_progress({"tag": "t", "transpose": -7.0},
                                            seconds=1.0, elapsed=1.0))

    def test_the_report_counts_what_happened(self):
        from tools.svc_batch import run_batch
        rep = run_batch([self._job("a"), self._job("b")],
                        load_models=lambda: {}, convert_one=lambda job, m: None)
        self.assertEqual(rep["converted"], ["a", "b"])
        self.assertEqual(rep["n_total"], 2)


class ConversionConsistencyTests(unittest.TestCase):
    """同じ test set の中で変換条件が揃っていること（M5）。

    **実際に混ざりました。** 26 clip のうち 15 本を `svc_convert.py`（chunk 20 秒）、
    11 本を `svc_batch.py`（当時の既定 10 秒）で作ってしまい、**20 秒の clip では
    chunk 1 個と 2 個で境界処理が変わります**。比較の土台としては使えません。

    条件は各 clip の `*_convert.json` に残るので、**測る前に揃っているかを確かめます。**
    """

    def _rec(self, **kw):
        base = {"ckpt": "c.pt", "spk_id": 22, "num_steps": 1, "device": "cuda",
                "chunk_sec": 20.0}
        base.update(kw)
        return base

    def test_a_consistent_set_passes(self):
        from tools.svc_batch import check_consistent
        r = check_consistent([self._rec(), self._rec(), self._rec()])
        self.assertTrue(r["consistent"])

    def test_a_differing_chunk_size_is_caught(self):
        from tools.svc_batch import check_consistent
        r = check_consistent([self._rec(chunk_sec=20.0), self._rec(chunk_sec=10.0)])
        self.assertFalse(r["consistent"])
        self.assertIn("chunk_sec", r["differing"])

    def test_a_differing_checkpoint_is_caught(self):
        from tools.svc_batch import check_consistent
        r = check_consistent([self._rec(ckpt="a.pt"), self._rec(ckpt="b.pt")])
        self.assertIn("ckpt", r["differing"])

    def test_a_differing_device_is_caught(self):
        # CPU と GPU は bit 一致しない。同じ set に混ぜない。
        from tools.svc_batch import check_consistent
        r = check_consistent([self._rec(device="cpu"), self._rec(device="cuda")])
        self.assertIn("device", r["differing"])

    def test_per_clip_fields_are_not_compared(self):
        # transpose や start は clip ごとに違って当然。ここを条件違いと呼ばない。
        from tools.svc_batch import check_consistent
        r = check_consistent([self._rec(transpose=0, start=0.0),
                              self._rec(transpose=12, start=42.0)])
        self.assertTrue(r["consistent"])

    def test_a_missing_field_is_reported_not_ignored(self):
        # 古い形式の json（chunk_sec を持たない）が混ざったら気づけること。
        from tools.svc_batch import check_consistent
        rec = self._rec()
        del rec["chunk_sec"]
        r = check_consistent([self._rec(), rec])
        self.assertFalse(r["consistent"])
        self.assertIn("chunk_sec", r["differing"])


class PitchMetricsTests(unittest.TestCase):
    """F0 追従と V/UV（M5 ゴール 2）。

    **変換済みの WAV から測ります。** `m3_verify.py` は自分で変換してしまうので、
    既に作った test set（両システム分）には使えません。**同じ出力を両系で測る**必要が
    あります。

    F0 抽出器は引数で受け取るので、このテストは重いモデルを使いません。
    """

    def test_correlation_is_one_for_identical_pitch(self):
        from tools.pitch_metrics import pitch_report
        f0 = np.array([100.0, 110.0, 120.0, 130.0], dtype=np.float32)
        uv = np.ones(4, dtype=np.float32)
        r = pitch_report((f0, uv), (f0, uv))
        self.assertAlmostEqual(r["f0_corr"], 1.0, places=6)
        self.assertAlmostEqual(r["median_abs_semitones"], 0.0, places=6)

    def test_a_constant_octave_shift_shows_as_twelve_semitones(self):
        from tools.pitch_metrics import pitch_report
        f0 = np.array([100.0, 110.0, 120.0, 130.0], dtype=np.float32)
        uv = np.ones(4, dtype=np.float32)
        r = pitch_report((f0, uv), (f0 * 2, uv))
        self.assertAlmostEqual(r["median_abs_semitones"], 12.0, places=4)
        self.assertAlmostEqual(r["f0_corr"], 1.0, places=6)

    def test_an_expected_transpose_is_removed_before_comparing(self):
        # 男性 source は +12 半音で変換している。それを誤差として数えない。
        from tools.pitch_metrics import pitch_report
        f0 = np.array([100.0, 110.0, 120.0, 130.0], dtype=np.float32)
        uv = np.ones(4, dtype=np.float32)
        r = pitch_report((f0, uv), (f0 * 2, uv), transpose=12.0)
        self.assertAlmostEqual(r["median_abs_semitones"], 0.0, places=4)

    def test_unvoiced_frames_are_excluded_from_the_pitch_error(self):
        # 無声フレームの F0 は意味を持たない。混ぜると誤差が壊れる。
        from tools.pitch_metrics import pitch_report
        f0a = np.array([100.0, 0.0, 120.0], dtype=np.float32)
        f0b = np.array([100.0, 999.0, 120.0], dtype=np.float32)
        uv = np.array([1.0, 0.0, 1.0], dtype=np.float32)
        r = pitch_report((f0a, uv), (f0b, uv))
        self.assertAlmostEqual(r["median_abs_semitones"], 0.0, places=6)

    def test_uv_agreement_counts_matching_frames(self):
        from tools.pitch_metrics import pitch_report
        f0 = np.array([100.0, 100.0, 100.0, 100.0], dtype=np.float32)
        a = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        r = pitch_report((f0, a), (f0, b))
        self.assertAlmostEqual(r["uv_agree"], 0.75)

    def test_length_mismatch_is_trimmed_to_the_shorter(self):
        # 系ごとに端数フレームが違う。**黙って伸ばさず、短いほうへ揃える。**
        from tools.pitch_metrics import pitch_report
        f0a = np.array([100.0, 110.0, 120.0], dtype=np.float32)
        f0b = np.array([100.0, 110.0], dtype=np.float32)
        r = pitch_report((f0a, np.ones(3, np.float32)), (f0b, np.ones(2, np.float32)))
        self.assertEqual(r["n_frames"], 2)

    def test_no_voiced_frames_returns_none_rather_than_zero(self):
        # 無声だけの clip で 0 を返すと「完璧に一致」と読めてしまう。
        from tools.pitch_metrics import pitch_report
        z = np.zeros(4, dtype=np.float32)
        r = pitch_report((z, z), (z, z))
        self.assertIsNone(r["median_abs_semitones"])
        self.assertIsNone(r["f0_corr"])


class FailureTaxonomyTests(unittest.TestCase):
    """failure sample の分類（M5 ゴール 4、[評価計画](doc/svc-evaluation.md) 7 節）。

    **failure を除外せず、分類して残します。** 悪い clip を捨てると平均は良くなりますが、
    何が壊れるのかが分からなくなります。

    **機械的に判定できるものだけを扱います。** 「こもっている」「不自然」のような聴感は
    ここでは扱いません（blind test の担当）。判定に使う数値は既に測ってあるものです。
    """

    def _clip(self, **kw):
        base = {"tag": "unseen00", "uv_agree": 0.99, "median_abs_semitones": 0.1,
                "matched_ratio": 0.7, "cer_excess": 0.1, "centroid_ratio": 1.0,
                "peak": 0.5, "finite": True}
        base.update(kw)
        return base

    def test_a_clean_clip_has_no_categories(self):
        from tools.failure_taxonomy import classify
        self.assertEqual(classify(self._clip()), [])

    def test_a_silent_output_is_flagged(self):
        from tools.failure_taxonomy import classify
        self.assertIn("silence", classify(self._clip(peak=1e-5)))

    def test_a_non_finite_output_is_flagged(self):
        from tools.failure_taxonomy import classify
        self.assertIn("nonfinite", classify(self._clip(finite=False)))

    def test_an_octave_error_is_flagged_as_pitch(self):
        from tools.failure_taxonomy import classify
        self.assertIn("pitch", classify(self._clip(median_abs_semitones=11.8)))

    def test_a_small_pitch_deviation_is_not_flagged(self):
        from tools.failure_taxonomy import classify
        self.assertNotIn("pitch", classify(self._clip(median_abs_semitones=0.4)))

    def test_bad_voicing_agreement_is_flagged(self):
        from tools.failure_taxonomy import classify
        self.assertIn("voicing", classify(self._clip(uv_agree=0.70)))

    def test_many_lost_onsets_are_flagged_as_timing(self):
        from tools.failure_taxonomy import classify
        self.assertIn("timing", classify(self._clip(matched_ratio=0.35)))

    def test_a_large_cer_excess_is_flagged_as_content(self):
        from tools.failure_taxonomy import classify
        self.assertIn("content", classify(self._clip(cer_excess=0.6)))

    def test_a_dark_output_is_flagged_as_timbre(self):
        # 上限に対して明るさが大きく外れている（M3 で耳で分かった劣化）。
        from tools.failure_taxonomy import classify
        self.assertIn("timbre", classify(self._clip(centroid_ratio=0.5)))

    def test_a_bright_output_is_also_flagged(self):
        # 明るすぎるのも外れ。符号つきで見ると打ち消し合う。
        from tools.failure_taxonomy import classify
        self.assertIn("timbre", classify(self._clip(centroid_ratio=1.8)))

    def test_missing_values_do_not_produce_a_category(self):
        # 測れなかった軸を「失敗」と数えない（CER は 8 clip で判別不能だった）。
        from tools.failure_taxonomy import classify
        self.assertEqual(classify(self._clip(cer_excess=None,
                                             median_abs_semitones=None)), [])

    def test_several_categories_can_apply_at_once(self):
        from tools.failure_taxonomy import classify
        got = classify(self._clip(peak=1e-6, uv_agree=0.5))
        self.assertIn("silence", got)
        self.assertIn("voicing", got)

    def test_summarise_counts_clips_per_category(self):
        from tools.failure_taxonomy import summarise
        rows = [{"tag": "a", "categories": ["pitch"]},
                {"tag": "b", "categories": ["pitch", "timbre"]},
                {"tag": "c", "categories": []}]
        s = summarise(rows)
        self.assertEqual(s["counts"]["pitch"], 2)
        self.assertEqual(s["n_clean"], 1)
        self.assertEqual(s["n_total"], 3)

    def test_summarise_keeps_the_tags_for_each_category(self):
        # 「どの clip か」が分からないと後から聴き直せない。
        from tools.failure_taxonomy import summarise
        rows = [{"tag": "a", "categories": ["pitch"]}, {"tag": "b", "categories": ["pitch"]}]
        self.assertEqual(summarise(rows)["tags"]["pitch"], ["a", "b"])


class SvcDefaultStepsTests(unittest.TestCase):
    """SVC 推論の既定 step 数（2026-09-01 の掃引で決定）。

    **SVC 経路だけを変えます。** 未知 source 20 clip の掃引で、事前登録した規則
    （話者類似度を最大化、内容 cos の低下 0.02 以内、同点なら小さい step）が
    **16** を選びました。回復率 58.9% -> 68.6%、内容 cos −0.019、flow 時間は 2.1 倍。

    **SVS 経路（`infer_mel`）は触りません。** 今回の測定は SVC 経路のみで、SVS の
    1 step 品質は別途検証済みだからです。**測っていない経路の既定を変えないこと。**
    """

    def test_the_test_set_uses_the_swept_transpose(self):
        """**掃引で決めた値と test set の値がずれないこと。**

        2026-09-16 の掃引（natsume 8 clip、+0 / +7 / +10 / +12）で、事前登録した規則が
        **+7** を選びました。**それまでの +12 は両方の軸で劣ります** ―― 話者類似度の
        回復率が 90.8% -> 81.6%、CER の上限との差が +12.5 -> **+27.0 点**（制約は 17.4 点）。
        """
        from tools.m5_testset import MALE_TRANSPOSE
        from tools.svc_defaults import SVC_TRANSPOSE_LOW_VOICE
        self.assertEqual(float(MALE_TRANSPOSE), float(SVC_TRANSPOSE_LOW_VOICE))

    def test_the_swept_transpose_is_the_value_the_rule_chose(self):
        from tools.svc_defaults import SVC_TRANSPOSE_LOW_VOICE
        self.assertEqual(SVC_TRANSPOSE_LOW_VOICE, 7.0)

    def test_svc_inference_defaults_to_the_chosen_steps(self):
        import inspect

        from infer import infer_svc_mel
        from tools.svc_defaults import SVC_NUM_STEPS
        self.assertEqual(
            inspect.signature(infer_svc_mel).parameters["num_steps"].default, SVC_NUM_STEPS)

    def test_the_chosen_value_is_what_the_sweep_selected(self):
        from tools.svc_defaults import SVC_NUM_STEPS
        self.assertEqual(SVC_NUM_STEPS, 16)

    def test_the_svs_path_is_left_alone(self):
        # SVS の既定を巻き込むと、測っていない経路の挙動を変えてしまう。
        import inspect

        from infer import infer_mel
        self.assertEqual(inspect.signature(infer_mel).parameters["num_steps"].default, 10)

    def test_the_svc_clis_use_the_same_default(self):
        # ツールごとに既定が違うと、どの step で測ったのか分からなくなる。
        import re
        from pathlib import Path

        from tools.svc_defaults import SVC_NUM_STEPS
        for name in ("svc_convert.py", "svc_batch.py", "m3_verify.py"):
            src = (Path(__file__).parent / "tools" / name).read_text(encoding="utf-8")
            m = re.search(r'"--num-steps".*?default=(\w+)', src, re.S)
            self.assertIsNotNone(m, f"{name} に --num-steps が無い")
            self.assertEqual(m.group(1), "SVC_NUM_STEPS",
                             f"{name} が既定を直書きしている（{m.group(1)}）")
        self.assertEqual(SVC_NUM_STEPS, 16)


class SvcGanSmokeTests(unittest.TestCase):
    """SVC 経路の GAN 配線が smoke で踏まれること。

    **GAN 付き SVC は一度も動かしていない経路です。** smoke は SVS 側の GAN しか踏んで
    おらず、SVC は `gan.enabled: false` のままでした。**vast.ai で数時間を投じる前に、
    配線が通ることを手元で確かめます。**

    ここでは smoke の構成だけを検査します（実際の学習は smoke 本体が回す）。
    """

    def _smoke_src(self):
        from pathlib import Path
        return (Path(__file__).parent / "tools" / "smoke" / "run_smoke.py").read_text(
            encoding="utf-8")

    def test_a_gan_enabled_svc_config_is_written(self):
        # svc.yaml とは別に、GAN を有効にした config を書くこと。
        src = self._smoke_src()
        self.assertIn("svc_gan.yaml", src)

    def test_the_svc_gan_config_turns_gan_on(self):
        src = self._smoke_src()
        self.assertRegex(src, r'svc_gan\["gan"\]\.update\([^)]*enabled=True')

    def test_the_gan_starts_early_enough_for_a_short_smoke(self):
        # gan_start_step が既定の 2000 のままだと、20 step の smoke では GAN 経路を
        # 一度も通らずに「通った」ことになる。
        src = self._smoke_src()
        self.assertRegex(src, r'svc_gan\["gan"\]\.update\([^)]*gan_start_step=\d')

    def test_there_is_a_stage_running_svc_with_gan(self):
        src = self._smoke_src()
        self.assertIn("svc-train-gan", src)

    def test_the_stage_is_registered_in_the_stage_list(self):
        # 関数を書いても一覧に足さなければ走らない。
        src = self._smoke_src()
        self.assertRegex(src, r'\("svc-train-gan",\s*st_svc_train_gan')


class VastBootstrapTargetTests(unittest.TestCase):
    """bootstrap が指すリポジトリとブランチが、2 つのファイルで一致していること。

    **実際に壊れました。** `feature/svc` を main へ merge して削除した日、
    `tools/vast.py` の `BOOTSTRAP_URL` と `tools/vast_bootstrap.sh` の既定が
    **どちらも消えたブランチと旧リポジトリ名を指したまま**残り、インスタンスの
    立ち上げが失敗する状態になっていました。**同じ事実が 2 か所にあるので、
    片方だけ直す事故も起きます。**
    """

    ROOT = Path(__file__).parent

    def _url_target(self):
        """`BOOTSTRAP_URL` から owner/repo/branch を取り出す。"""
        import re

        from tools.vast import BOOTSTRAP_URL
        # **ブランチ名に `/` が入り得る**ので、`/tools/` までを branch として切る
        # （`[^/]+` で採ると `feature/svc` が `feature` になり、検査が空振りします）。
        m = re.match(r"https://raw\.githubusercontent\.com/([^/]+)/([^/]+)/(.+)/tools/",
                     BOOTSTRAP_URL)
        self.assertIsNotNone(m, f"raw URL の形が想定と違う: {BOOTSTRAP_URL}")
        return m.group(1), m.group(2), m.group(3)

    def _script_target(self):
        """`vast_bootstrap.sh` の `REPO` / `BRANCH` の既定を取り出す。"""
        import re
        text = (self.ROOT / "tools/vast_bootstrap.sh").read_text(encoding="utf-8")
        repo = re.search(r'^REPO="\$\{REPO:-https://github\.com/([^/]+)/([^.]+)\.git\}"',
                         text, re.M)
        branch = re.search(r'^BRANCH="\$\{BRANCH:-([^}]+)\}"', text, re.M)
        self.assertIsNotNone(repo, "REPO の既定が読めない")
        self.assertIsNotNone(branch, "BRANCH の既定が読めない")
        return repo.group(1), repo.group(2), branch.group(1)

    def test_the_two_files_agree_on_repository_and_branch(self):
        self.assertEqual(self._url_target(), self._script_target())

    def test_the_bootstrap_does_not_point_at_a_deleted_branch(self):
        # `feature/svc` は 2026-09-18 に main へ merge して削除済み。
        for owner, repo, branch in (self._url_target(), self._script_target()):
            self.assertNotEqual(branch, "feature/svc",
                                f"{owner}/{repo} の branch が削除済みを指している")

    def test_the_bootstrap_points_at_the_current_repository_name(self):
        # リポジトリは 2026-09-17 に LeapSinger -> LeapSVC へ改名した。
        # **raw.githubusercontent.com は改名を追随しない**ので、旧名では 404 になる。
        for owner, repo, _ in (self._url_target(), self._script_target()):
            self.assertEqual(repo, "LeapSVC", f"{owner}/{repo} は旧リポジトリ名")


class VastCliResolutionTests(unittest.TestCase):
    """vastai CLI の呼び出し方（M5 の GAN 実験の直前に踏んだ）。

    **`vastai.exe` は uv の shim で、実行時に別の Python を参照します。** その Python が
    壊れていると `ImportError: DLL load failed while importing _socket` で落ち、
    `uv tool install --reinstall` でも直りません（壊れているのは shim の側だから）。

    **ツール環境の Python が生きていれば `-m vastai.cli.main` で回避できます。**
    実際にそうなっていたので、その経路を持たせます。
    """

    def test_it_prefers_a_working_launcher(self):
        from tools.vast import _vastai_cmd
        cmd = _vastai_cmd(which=lambda n: "/usr/bin/vastai", probe=lambda c: True)
        self.assertEqual(cmd, ["/usr/bin/vastai"])

    def test_a_broken_launcher_falls_back_to_the_tool_interpreter(self):
        from tools.vast import _vastai_cmd
        cmd = _vastai_cmd(which=lambda n: "/broken/vastai",
                          probe=lambda c: "python" in c[0],
                          tool_python="/tools/vastai/python.exe")
        self.assertEqual(cmd, ["/tools/vastai/python.exe", "-m", "vastai.cli.main"])

    def test_it_fails_loudly_when_nothing_works(self):
        # 黙って別の経路を試し続けず、何が壊れているかを言って止まる。
        from tools.vast import _vastai_cmd
        with self.assertRaises(SystemExit) as cm:
            _vastai_cmd(which=lambda n: None, probe=lambda c: False, tool_python=None)
        self.assertIn("vastai", str(cm.exception))

    def test_the_child_environment_drops_the_parent_interpreter_vars(self):
        """**`uv run` 配下から別の Python を起動すると壊れます。**

        `uv run` は `PYTHONHOME` を親（3.13）に向けて設定し、それが子へ継承されるので、
        ツール環境の Python（3.14）が **3.13 の標準ライブラリを読みに行き**
        `ImportError: DLL load failed while importing _socket` で落ちます。
        **親のインタプリタを指す変数を落としてから起動すること。**
        """
        from tools.vast import _child_env
        env = _child_env({"PYTHONHOME": "/py313", "PYTHONPATH": "/py313/lib",
                          "VIRTUAL_ENV": "/repo/.venv", "PATH": "/usr/bin"})
        for k in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"):
            self.assertNotIn(k, env)
        self.assertEqual(env["PATH"], "/usr/bin")

    def test_unrelated_variables_survive(self):
        # API token などを落とすと動かなくなる。落とすのは干渉するものだけ。
        from tools.vast import _child_env
        env = _child_env({"VAST_API_KEY": "x", "PYTHONHOME": "/py"})
        self.assertEqual(env["VAST_API_KEY"], "x")

    def test_the_probe_actually_runs_the_candidate(self):
        # 存在確認だけでは足りない。**実際に起動できるか**を見る（今回の壊れ方は起動時）。
        from tools.vast import _vastai_cmd
        probed = []
        _vastai_cmd(which=lambda n: "/usr/bin/vastai",
                    probe=lambda c: probed.append(c) or True)
        self.assertEqual(probed, [["/usr/bin/vastai"]])



class BlindListenPageTests(unittest.TestCase):
    """ブラウザで聴いて投票するページ。**blind を崩さないことが最優先の契約。**"""

    ROWS = [{"pair": "pair00", "clip": "unseen08"},
            {"pair": "pair01", "clip": "holdout00"},
            {"pair": "pair02", "clip": "unseen17"}]

    def test_every_pair_appears(self):
        from tools.blind_test import listen_page
        html = listen_page(self.ROWS)
        for r in self.ROWS:
            self.assertIn(r["pair"], html)

    def test_both_sides_are_reachable_as_audio(self):
        from tools.blind_test import listen_page
        html = listen_page(self.ROWS)
        self.assertIn("audio/pair00_A.wav", html)
        self.assertIn("audio/pair00_B.wav", html)
        self.assertIn("paired/pair00_AB.wav", html)

    def test_key_rows_are_refused(self):
        # key.json の行を渡すと **答えがページに埋まる**。受け取らないこと。
        from tools.blind_test import listen_page
        with self.assertRaises(ValueError):
            listen_page([{"pair": "pair00", "clip": "unseen08",
                          "A": "leapsvc", "B": "seedvc"}])

    def test_system_names_never_reach_the_page(self):
        from tools.blind_test import listen_page
        html = listen_page(self.ROWS).lower()
        for name in ("leapsvc", "seedvc", "seed-vc"):
            self.assertNotIn(name, html)

    def test_sheet_order_is_preserved(self):
        # prepare が shuffle 済み。ページが並べ替えると曲順の癖が戻る。
        from tools.blind_test import listen_page
        html = listen_page(self.ROWS)
        self.assertLess(html.index("pair00"), html.index("pair01"))
        self.assertLess(html.index("pair01"), html.index("pair02"))

    def test_clip_names_are_shown_for_bookkeeping(self):
        from tools.blind_test import listen_page
        html = listen_page(self.ROWS)
        self.assertIn("unseen08", html)

    def test_empty_rows_are_refused(self):
        from tools.blind_test import listen_page
        with self.assertRaises(ValueError):
            listen_page([])

    def test_exported_csv_header_matches_the_sheet(self):
        # ページが吐く CSV を tally がそのまま読めること。
        from tools.blind_test import SHEET_HEADER, listen_page
        self.assertIn(SHEET_HEADER, listen_page(self.ROWS))

    def test_existing_votes_are_carried_into_the_page(self):
        # 途中まで書いたシートから再開できること（26 ペアを一度で聴き切るとは限らない）。
        from tools.blind_test import listen_page
        html = listen_page([{"pair": "pair00", "clip": "unseen08", "vote": "A"},
                            {"pair": "pair01", "clip": "holdout00", "vote": ""}])
        self.assertIn('"vote": "A"', html.replace("'", '"'))

    def test_bad_vote_in_the_sheet_is_refused(self):
        from tools.blind_test import listen_page
        with self.assertRaises(ValueError):
            listen_page([{"pair": "pair00", "clip": "unseen08", "vote": "X"}])

    def test_reference_audio_is_embedded(self):
        from tools.blind_test import listen_page
        html = listen_page(self.ROWS,
                           references=[{"label": "target（波音リツ）",
                                        "path": "context/target_ref.wav"}])
        self.assertIn("context/target_ref.wav", html)
        self.assertIn("波音リツ", html)

    def test_per_pair_source_is_embedded(self):
        from tools.blind_test import listen_page
        rows = [{"pair": "pair00", "clip": "unseen08",
                 "source": "context/pair00_source.wav"}]
        self.assertIn("context/pair00_source.wav", listen_page(rows))

    def test_paths_outside_the_blind_directory_are_refused(self):
        # 元ディレクトリを直接指すと **パスに system 名が出て** blind が崩れる。
        from tools.blind_test import listen_page
        with self.assertRaises(ValueError):
            listen_page(self.ROWS,
                        references=[{"label": "t", "path": "out/m5/leapsvc/_target_ref.wav"}])
        with self.assertRaises(ValueError):
            listen_page([{"pair": "pair00", "clip": "c",
                          "source": "../leapsvc/x_source.wav"}])

    def test_the_ceiling_is_refused_as_a_reference(self):
        # 上限（GT mel -> ボコーダー）は **LeapSVC 自身のボコーダーの音**。参照として
        # 聴かせると、その癖で A / B のどちらが LeapSVC かを当てられてしまう。
        from tools.blind_test import listen_page
        with self.assertRaises(ValueError):
            listen_page(self.ROWS,
                        references=[{"label": "上限", "path": "context/x_vocoder_only.wav"}])

    def test_the_pre_registered_criterion_is_stated(self):
        # 事前登録した基準は **自然さ**であって target 類似度ではない。
        from tools.blind_test import listen_page
        html = listen_page(self.ROWS)
        self.assertIn("自然", html)

    def test_source_lookup_is_exact_not_a_prefix(self):
        # `unseen1` が `unseen10` に当たると、**別 clip の変換元を聴かせる**ことになる。
        import tempfile
        from pathlib import Path

        from tools.blind_test import find_source
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "song__unseen10_source.wav").write_bytes(b"")
            (root / "song__unseen1_source.wav").write_bytes(b"")
            self.assertEqual(find_source(root, "unseen1").name, "song__unseen1_source.wav")
            self.assertEqual(find_source(root, "unseen10").name, "song__unseen10_source.wav")

    def test_missing_source_returns_none(self):
        import tempfile
        from pathlib import Path

        from tools.blind_test import find_source
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(find_source(Path(d), "unseen1"))

    def test_ambiguous_source_is_refused(self):
        # 同じ clip の変換元が 2 つあるなら、どちらを聴かせるかは決められない。
        import tempfile
        from pathlib import Path

        from tools.blind_test import find_source
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "a__unseen1_source.wav").write_bytes(b"")
            (root / "b__unseen1_source.wav").write_bytes(b"")
            with self.assertRaises(ValueError):
                find_source(root, "unseen1")

    def test_sheet_written_by_prepare_round_trips(self):
        # **prepare が書くヘッダには説明が付く**（`vote  # vote に ...`）。列名をそのまま
        # 引くと **全件が未記入**になり、票が丸ごと消える。実際に起きた。
        import tempfile
        from pathlib import Path

        from tools.blind_test import SHEET_HEADER, read_sheet
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "sheet.csv"
            p.write_text(SHEET_HEADER + "\npair00,unseen08,B\npair01,holdout00,tie\n",
                         encoding="utf-8")
            rows = read_sheet(p)
            self.assertEqual([r["vote"] for r in rows], ["B", "tie"])
            self.assertEqual([r["pair"] for r in rows], ["pair00", "pair01"])

    def test_plain_vote_header_also_works(self):
        import tempfile
        from pathlib import Path

        from tools.blind_test import read_sheet
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "sheet.csv"
            p.write_text("pair,clip,vote\npair00,unseen08,A\n", encoding="utf-8")
            self.assertEqual(read_sheet(p)[0]["vote"], "A")

    def test_a_column_merely_containing_vote_is_not_the_vote_column(self):
        # `note_vote` を票と読むと別の値が票になる。**黙って空を返すのも不可** ――
        # 今回 26 票を失ったのがまさにその壊れ方なので、無ければ落とす。
        import tempfile
        from pathlib import Path

        from tools.blind_test import read_sheet
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "sheet.csv"
            p.write_text("pair,clip,note_vote\npair00,unseen08,A\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_sheet(p)

    def test_missing_vote_column_is_refused(self):
        import tempfile
        from pathlib import Path

        from tools.blind_test import read_sheet
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "sheet.csv"
            p.write_text("pair,clip\npair00,unseen08\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_sheet(p)


class BlindWobbleQuestionTests(unittest.TestCase):
    """**名指しされた defect を直接聞く質問。**

    2026-09-13 の聴取で 6 本中 5 本が「音量が揺れる」を挙げました。preference で
    聞き直すと、**差が「良さ」に化けて記録されます**（選択肢に無い defect が近い
    ラベルに化けた、というのは実際に踏んだ失敗）。
    """

    ROWS = [{"pair": "pair00", "clip": "unseen17"},
            {"pair": "pair01", "clip": "unseen04"}]

    def test_the_wobble_question_is_registered(self):
        from tools.blind_test import QUESTIONS
        self.assertIn("wobble", QUESTIONS)

    def test_the_wobble_question_asks_about_level_not_quality(self):
        from tools.blind_test import listen_page
        html = listen_page(self.ROWS, question="wobble")
        self.assertIn("音量", html)
        # 良し悪しを一緒に聞くと、**preference の交絡（明るさ・大きさ）**が戻る。
        self.assertNotIn("どちらが良い", html)
        self.assertNotIn("自然さ", html)

    def test_the_wobble_question_needs_no_target_reference(self):
        # target に似ているかは聞かない。**基準が要らない質問**なので、
        # 参照なしで作れること自体が契約。
        from tools.blind_test import listen_page
        self.assertIn("音量", listen_page(self.ROWS, question="wobble"))

    def test_the_wobble_votes_do_not_share_storage_with_the_others(self):
        # **同じ localStorage を使うと、前の質問の票が黙って埋まります。**
        import re

        from tools.blind_test import listen_page
        key = re.compile(r'KEY\s*=\s*"([^"]+)"')
        got = {q: key.search(listen_page(
            self.ROWS, question=q,
            references=[{"label": "target 本人（波音リツ）",
                         "path": "context/target_ref.wav"}])).group(1)
            for q in ("preference", "similarity", "wobble")}
        self.assertEqual(len(set(got.values())), 3, got)

    def test_the_wobble_question_writes_its_own_sheet(self):
        from tools.blind_test import QUESTIONS
        sheets = {q: QUESTIONS[q]["sheet"] for q in ("preference", "similarity", "wobble")}
        self.assertEqual(len(set(sheets.values())), 3, sheets)

    def test_tally_accepts_the_wobble_question(self):
        # 集計が質問を知らないと、**別の質問の票を preference として読みます。**
        from tools.blind_test import tally
        sheet = [{"pair": "pair00", "clip": "unseen17", "vote": "A",
                  "A": "v31", "B": "v32"},
                 {"pair": "pair01", "clip": "unseen04", "vote": "B",
                  "A": "v32", "B": "v31"}]
        r = tally(sheet, question="wobble")
        self.assertEqual(r["question"], "wobble")


class BlindSimilarityPageTests(unittest.TestCase):
    """**質問が違えば別の test です。** preference の票が混ざらないことが最優先の契約。"""

    ROWS = [{"pair": "pair00", "clip": "unseen08"},
            {"pair": "pair01", "clip": "holdout00"}]
    REF = [{"label": "target 本人（波音リツ）", "path": "context/target_ref.wav"}]

    def test_the_similarity_question_asks_about_the_target(self):
        # ①の質問は「どちらが target 本人に似ているか」1 つだけ。
        from tools.blind_test import listen_page
        html = listen_page(self.ROWS, question="similarity", references=self.REF)
        self.assertIn("似て", html)
        self.assertIn("波音リツ", html)

    def test_the_similarity_question_does_not_ask_which_is_better(self):
        # 良し悪し・自然さを一緒に聞くと、**前回と同じ交絡（明るさ）**が入る。
        from tools.blind_test import listen_page
        html = listen_page(self.ROWS, question="similarity", references=self.REF)
        self.assertNotIn("自然さ", html)
        self.assertNotIn("どちらが良い", html)

    def test_the_similarity_page_requires_a_target_reference(self):
        # 「似ている」は基準が無ければ判断できない。**黙って基準なしで聴かせない。**
        from tools.blind_test import listen_page
        with self.assertRaises(ValueError):
            listen_page(self.ROWS, question="similarity")

    def test_the_two_questions_do_not_share_vote_storage(self):
        # **同じ localStorage を使うと、前回の preference の票が黙って埋まります。**
        # 例外は出ず、「もう答えてある」ように見えるのが壊れ方。
        import re

        from tools.blind_test import listen_page
        pref = listen_page(self.ROWS)
        sim = listen_page(self.ROWS, question="similarity", references=self.REF)
        key = re.compile(r'KEY\s*=\s*"([^"]+)"')
        self.assertIsNotNone(key.search(pref))
        self.assertIsNotNone(key.search(sim))
        self.assertNotEqual(key.search(pref).group(1), key.search(sim).group(1))

    def test_preference_is_still_the_default(self):
        # M5 のページを再現できること（既定を変えると過去の成果物が作り直せない）。
        from tools.blind_test import listen_page
        self.assertEqual(listen_page(self.ROWS),
                         listen_page(self.ROWS, question="preference"))

    def test_an_unknown_question_is_refused(self):
        # 黙って preference に落ちると、**別の質問を聞いたつもりの票**が貯まる。
        from tools.blind_test import listen_page
        with self.assertRaises(ValueError):
            listen_page(self.ROWS, question="なんとなく", references=self.REF)

    def test_the_page_names_the_sheet_it_writes(self):
        # 書き出し先が固定だと、**preference の sheet.csv を上書き**してしまう。
        from tools.blind_test import listen_page
        html = listen_page(self.ROWS, question="similarity", references=self.REF,
                           sheet_path="out/m5/blind_sim/sheet.csv")
        self.assertIn("out/m5/blind_sim/sheet.csv", html)

    def test_system_names_never_reach_the_similarity_page(self):
        from tools.blind_test import listen_page
        html = listen_page(self.ROWS, question="similarity", references=self.REF).lower()
        for name in ("leapsvc", "seedvc", "seed-vc"):
            self.assertNotIn(name, html)

    def test_tally_records_which_side_was_chosen(self):
        # **位置の偏りを出さない集計は、系の差と取り違えます。** 判別できないとき、
        # 人は「最後に聴いたほう」を選びます（A -> B の並びでは B）。実際に起きた。
        from tools.blind_test import tally
        sheet = [{"pair": f"pair{i:02d}", "clip": f"c{i}", "vote": "B",
                  "A": "x", "B": "y"} for i in range(4)]
        rep = tally(sheet, question="similarity")
        self.assertEqual(rep["sides"], {"A": 0, "B": 4})

    def test_a_side_bias_is_not_read_as_a_system_difference(self):
        # 全票が B でも、B に居た系が入れ替わっていれば勝敗は割れる。**両方出すこと。**
        from tools.blind_test import tally
        sheet = [{"pair": "pair00", "clip": "c0", "vote": "B", "A": "x", "B": "y"},
                 {"pair": "pair01", "clip": "c1", "vote": "B", "A": "y", "B": "x"}]
        rep = tally(sheet)
        self.assertEqual(rep["sides"], {"A": 0, "B": 2})
        self.assertEqual(rep["wins"], {"x": 1, "y": 1})

    def test_side_bias_gets_its_own_p_value(self):
        # 系の p と混ぜない。**別の仮説**（位置の偏り）なので別に出す。
        from tools.blind_test import tally
        sheet = [{"pair": f"pair{i:02d}", "clip": f"c{i}", "vote": "B",
                  "A": "x" if i % 2 else "y", "B": "y" if i % 2 else "x"}
                 for i in range(6)]
        rep = tally(sheet)
        self.assertAlmostEqual(rep["p_side"], 0.03125, places=5)
        self.assertGreater(rep["p_two_sided"], 0.9)


    def test_tally_records_which_question_was_asked(self):
        # 質問を書かない result.json は、2 つ並ぶと**どちらの結果か分からなくなります**。
        from tools.blind_test import tally
        sheet = [{"pair": "pair00", "clip": "c0", "vote": "A", "A": "x", "B": "y"}]
        self.assertEqual(tally(sheet, question="similarity")["question"], "similarity")

    def test_tally_does_not_invent_a_question(self):
        # 「記録が無い」と「preference だった」を混同しないこと。
        from tools.blind_test import tally
        sheet = [{"pair": "pair00", "clip": "c0", "vote": "A", "A": "x", "B": "y"}]
        self.assertIsNone(tally(sheet)["question"])


class CerBreakdownTests(unittest.TestCase):
    """CER を言語ごとに分ける。**pooled の中央値は素材の混合を隠します。**

    実測（M5、26 clip）: 公表値は差 **+16.8 点**でしたが、言語別に割ると
    **日本語 +4.8 / 英語 +13.1 / ラテン語 +20.5 / イタリア語 +93.3（n=1）**でした。
    **上限自身が 23.2% 揃わないラテン語**は、同じ音を片仮名と平仮名で書き起こしただけの
    差を拾っています。**上限が不安定な素材で差を出してはいけません。**
    """

    @staticmethod
    def _clips(lang_ceil_conv):
        return [{"lang": lg, "cer_ceiling": ce, "cer_converted": cv,
                 "cer_excess_over_ceiling": cv - ce, "ceiling_unusable": False,
                 "asr_failed": False}
                for lg, ce, cv in lang_ceil_conv]

    def test_language_comes_from_the_piece_name(self):
        from tools.cer_breakdown import clip_language
        self.assertEqual(clip_language("holdout03", piece=None), "ja")
        self.assertEqual(clip_language("unseen00", piece="dona"), "la")
        self.assertEqual(clip_language("unseen01", piece="caro"), "it")
        self.assertEqual(clip_language("unseen02", piece="row"), "en")

    def test_an_unknown_piece_is_refused(self):
        # 黙って ja や unknown に落とすと、**読めない素材が読めたことになります**。
        from tools.cer_breakdown import clip_language
        with self.assertRaises(ValueError):
            clip_language("unseen00", piece="scales")

    def test_a_readable_language_reports_the_excess(self):
        from tools.cer_breakdown import breakdown
        rep = breakdown(self._clips([("ja", 0.02, 0.07)] * 5))
        g = rep["per_language"]["ja"]
        self.assertTrue(g["readable"])
        self.assertAlmostEqual(g["excess_median"], 0.05, places=6)

    def test_an_unstable_ceiling_gets_no_excess(self):
        # 上限は「同じ内容を自分のボコーダーに通したもの」。書き起こしが揃わない素材では、
        # **差は内容の劣化ではなく表記の揺れ**です。
        from tools.cer_breakdown import breakdown
        rep = breakdown(self._clips([("la", 0.23, 0.52)] * 6))
        g = rep["per_language"]["la"]
        self.assertFalse(g["readable"])
        self.assertIsNone(g["excess_median"])
        self.assertEqual(g["reason"], "ceiling_unstable")

    def test_too_few_clips_gets_no_excess(self):
        from tools.cer_breakdown import breakdown
        rep = breakdown(self._clips([("it", 0.067, 1.0)]))
        g = rep["per_language"]["it"]
        self.assertFalse(g["readable"])
        self.assertEqual(g["reason"], "too_few_clips")

    def test_a_transcription_that_blew_up_is_not_counted_as_intelligibility(self):
        # **CER > 100% は「もっと聞き取れない」ではなく書き起こしの暴走**です（挿入で
        # 長さが膨らむ）。実測でイタリア語が 909% / 1733% を出し、中央値 +897.7 点が
        # 「読める」判定で通りました。**上限側にしか門が無かったのが原因。**
        from tools.cer_breakdown import breakdown
        rep = breakdown(self._clips([("it", 0.044, 9.091), ("it", 0.0, 17.333),
                                     ("it", 0.044, 1.0)]))
        g = rep["per_language"]["it"]
        self.assertEqual(g["n_degenerate"], 3)
        self.assertIsNone(g["excess_median"])
        self.assertEqual(g["reason"], "no_clean_clips")

    def test_degenerate_clips_are_reported_not_dropped(self):
        # **黙って落とすと本数が減ったことに気づけません。**
        from tools.cer_breakdown import breakdown
        clips = self._clips([("en", 0.0, 0.0), ("en", 0.0, 0.025), ("en", 0.0, 0.101),
                             ("en", 0.0, 0.177), ("en", 0.0, 1.0), ("en", 0.0, 4.519)])
        g = breakdown(clips)["per_language"]["en"]
        self.assertEqual(g["n"], 6)
        self.assertEqual(g["n_degenerate"], 2)
        self.assertEqual(g["n_clean"], 4)
        # 中央値は健全な 4 本だけ（0.0 / 2.5 / 10.1 / 17.7 -> 6.3 点）
        self.assertAlmostEqual(g["excess_median"], 0.063, places=3)
        self.assertTrue(g["readable"])

    def test_a_group_with_no_degenerate_clips_keeps_every_clip(self):
        from tools.cer_breakdown import breakdown
        g = breakdown(self._clips([("ja", 0.0, 0.056), ("ja", 0.024, 0.095),
                                   ("ja", 0.0, 0.097)]))["per_language"]["ja"]
        self.assertEqual(g["n_degenerate"], 0)
        self.assertEqual(g["n_clean"], 3)


    def test_the_pooled_median_is_reported_as_not_readable(self):
        # **混合の中央値を品質として出さない。** 出すが、読めない印を必ず付ける。
        from tools.cer_breakdown import breakdown
        rep = breakdown(self._clips([("ja", 0.02, 0.07)] * 3 + [("la", 0.23, 0.52)] * 6))
        self.assertIn("pooled", rep)
        self.assertFalse(rep["pooled"]["readable"])
        self.assertIn("言語", rep["pooled"]["reason_ja"])

    def test_clips_the_tool_already_rejected_are_not_counted(self):
        from tools.cer_breakdown import breakdown
        clips = self._clips([("ja", 0.02, 0.07)] * 3)
        clips.append({"lang": "ja", "cer_ceiling": 1.0, "cer_converted": 0.2,
                      "cer_excess_over_ceiling": None, "ceiling_unusable": True,
                      "asr_failed": False})
        self.assertEqual(breakdown(clips)["per_language"]["ja"]["n"], 3)


class SvcConvertProvenanceTests(unittest.TestCase):
    """変換の記録。**上限（`*_vocoder_only.wav`）が比べられるかを json が自分で申告する。**

    上限は GT mel だけでなく **RMVPE の F0 と UV にも依存**します
    （`mel_to_wav(vocoder, feats["mel"], logf0, feats["uv"])`）。CUDA では両方とも
    既定で bit 再現しないので、**同じ V3 の 2 つの run で 26 clip 中 6 本の上限 CER が
    5 点を超えてずれました**（最大 88 点）。`convert.json` は**どのボコーダーを使ったかも
    記録していませんでした** ―― 上限が比較できない原因そのものです。
    """

    def test_the_vocoder_is_recorded(self):
        # ボコーダーが変わると上限が変わる。**記録が無いと後から判別できません。**
        from tools.svc_convert import provenance
        p = provenance(ckpt="a.pt", vocoder="checkpoints/nhv_v3_1.onnx",
                       device="cpu", deterministic=False, env={})
        self.assertEqual(p["vocoder"], "checkpoints/nhv_v3_1.onnx")

    def test_a_freshly_generated_ceiling_is_never_comparable(self):
        """**ボコーダーが確率的なので、作り直した上限は必ず変わります。**

        当初は CUDA の非決定性を疑いましたが、**CPU + `use_deterministic_algorithms`
        でも 2 回の run で上限の sha256 が一致しませんでした**。原因は NHVSing の ONNX に
        **seed 属性の無い `RandomNormalLike`** があることで、ONNX Runtime が毎回別の乱数を
        引きます。**device も決定的モードも関係ありません。**
        """
        from tools.svc_convert import provenance
        for device, det in (("cpu", True), ("cpu", False), ("cuda", True)):
            p = provenance(ckpt="a.pt", vocoder="v.onnx", device=device,
                           deterministic=det, env={"CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
            self.assertFalse(p["ceiling_comparable"], f"{device}/{det}")
            self.assertEqual(p["ceiling_source"], "generated")

    def test_a_reused_ceiling_is_comparable(self):
        # **1 度だけ作って共有する**のが唯一の対処。系をまたいで同じ上限を使う。
        from tools.svc_convert import provenance
        p = provenance(ckpt="a.pt", vocoder="v.onnx", device="cuda", deterministic=False,
                       env={}, ceiling_from="out/m5/ceiling")
        self.assertTrue(p["ceiling_comparable"])
        self.assertEqual(p["ceiling_source"], "out/m5/ceiling")

    def test_the_vocoder_being_stochastic_is_recorded(self):
        from tools.svc_convert import provenance
        p = provenance(ckpt="a.pt", vocoder="v.onnx", device="cpu",
                       deterministic=True, env={})
        self.assertTrue(p["vocoder_stochastic"])

    def test_deterministic_on_cuda_needs_the_cublas_env(self):
        # 環境変数が無いと torch が実行時に落ちる。**変換を 30 分走らせてから落ちないこと。**
        from tools.svc_convert import deterministic_env_error
        self.assertIsNotNone(deterministic_env_error("cuda", {}))
        self.assertIn("CUBLAS_WORKSPACE_CONFIG", deterministic_env_error("cuda", {}))

    def test_the_documented_cublas_values_pass(self):
        from tools.svc_convert import deterministic_env_error
        for v in (":4096:8", ":16:8"):
            self.assertIsNone(deterministic_env_error("cuda", {"CUBLAS_WORKSPACE_CONFIG": v}))

    def test_cpu_needs_no_env(self):
        from tools.svc_convert import deterministic_env_error
        self.assertIsNone(deterministic_env_error("cpu", {}))

    def test_a_wrong_cublas_value_is_refused(self):
        from tools.svc_convert import deterministic_env_error
        self.assertIsNotNone(deterministic_env_error("cuda", {"CUBLAS_WORKSPACE_CONFIG": "1"}))


class BlindAnchorTests(unittest.TestCase):
    """anchor（catch trial）。**「2 系が同一」と「判別できていない」を分ける。**

    ①（target 類似の blind）で、判定 4 票すべてが「後に聴いた側」に張り付き、評価者は
    「意味がないように思えます」と述べました。**anchor が無いと、この 2 つを区別できません。**

    anchor は **答えが分かっているペア**です。target 本人の録音 対 無関係な話者を混ぜておき、
    **そこを外したら「判別できていない」**、当てられたのに本番が拮抗するなら
    **「2 系が近い」**と読めます。
    """

    def test_anchor_rows_carry_the_expected_answer(self):
        from tools.blind_test import make_anchors
        rows = make_anchors([{"target": "context/t.wav", "foil": "context/f.wav"}], seed=0)
        self.assertEqual(len(rows), 1)
        self.assertIn(rows[0]["expected"], ("A", "B"))

    def test_the_anchor_side_is_randomised(self):
        # 常に A が正解だと、並びを覚えられます。
        from tools.blind_test import make_anchors
        pairs = [{"target": f"context/t{i}.wav", "foil": f"context/f{i}.wav"}
                 for i in range(12)]
        sides = {r["expected"] for r in make_anchors(pairs, seed=3)}
        self.assertEqual(sides, {"A", "B"})

    def test_the_expected_answer_never_reaches_the_page(self):
        # **正解がページに出たら catch trial の意味がありません。**
        from tools.blind_test import listen_page, make_anchors
        anchors = make_anchors([{"target": "context/t.wav", "foil": "context/f.wav"}], seed=0)
        rows = [{"pair": "anchor00", "clip": "anchor", "a": anchors[0]["a"],
                 "b": anchors[0]["b"]}]
        html = listen_page(rows, question="similarity",
                           references=[{"label": "target", "path": "context/t.wav"}])
        self.assertNotIn("expected", html)

    def test_anchor_material_gets_a_common_sample_rate(self):
        # anchor の素材は別コーパスから来る（実測でリツ 44.1k / 棗 48k / 鬼灯 96k）。
        # **片側だけ触ると帯域が「どちらが target か」の手がかりになります。**
        # 両側を同じ rate へ対称に落とす。
        from tools.blind_test import anchor_common_sr
        self.assertEqual(anchor_common_sr([48000, 44100]), 44100)
        self.assertEqual(anchor_common_sr([44100, 44100]), 44100)
        self.assertEqual(anchor_common_sr([96000, 48000]), 48000)

    def test_anchor_common_rate_needs_rates(self):
        from tools.blind_test import anchor_common_sr
        with self.assertRaises(ValueError):
            anchor_common_sr([])

    def test_anchor_scoring_separates_the_two_failures(self):
        from tools.blind_test import score_anchors
        # 全問正解 -> 課題は遂行できている
        good = score_anchors([{"pair": "anchor00", "expected": "A", "vote": "A"},
                              {"pair": "anchor01", "expected": "B", "vote": "B"}])
        self.assertEqual(good["n_correct"], 2)
        self.assertTrue(good["task_performed"])
        # 全問不正解 -> **本番の拮抗を「2 系が同一」と読んではいけない**
        bad = score_anchors([{"pair": "anchor00", "expected": "A", "vote": "B"},
                             {"pair": "anchor01", "expected": "B", "vote": "A"}])
        self.assertEqual(bad["n_correct"], 0)
        self.assertFalse(bad["task_performed"])

    def test_a_tie_on_an_anchor_counts_as_failing_it(self):
        # anchor は target 本人 対 無関係な話者。**引き分けは判別できていない印です。**
        from tools.blind_test import score_anchors
        r = score_anchors([{"pair": "anchor00", "expected": "A", "vote": "tie"}])
        self.assertEqual(r["n_correct"], 0)
        self.assertEqual(r["n_tie"], 1)

    def test_unanswered_anchors_are_not_counted_as_wrong(self):
        # **未記入と間違いを混同しない。**
        from tools.blind_test import score_anchors
        r = score_anchors([{"pair": "anchor00", "expected": "A", "vote": ""}])
        self.assertEqual(r["n_missing"], 1)
        self.assertIsNone(r["task_performed"])

    def test_scoring_refuses_rows_without_an_expected_answer(self):
        # 本番のペアを混ぜて採点すると、**正解率が薄まって門が効かなくなります。**
        from tools.blind_test import score_anchors
        with self.assertRaises(ValueError):
            score_anchors([{"pair": "pair00", "vote": "A"}])


class CerSummaryStatisticTests(unittest.TestCase):
    """**どの統計量かを名前で明示する。**

    `asr_cer.py` の `cer_excess_over_ceiling` は **中央値の差**（median(変換) −
    median(上限)）でしたが、`cer_breakdown.py` は **差の中央値**です。名前が同じで
    中身が違ったため、**両方を混ぜて引用しました**（実測で +12.7 対 +10.7、
    +25.5 対 +18.8 と食い違います）。歪んだ分布では一致しません。
    """

    CLIPS = [
        {"cer_ceiling": 0.00, "cer_converted": 0.10, "cer_excess_over_ceiling": 0.10,
         "ceiling_unusable": False, "asr_failed": False},
        {"cer_ceiling": 0.02, "cer_converted": 0.12, "cer_excess_over_ceiling": 0.10,
         "ceiling_unusable": False, "asr_failed": False},
        {"cer_ceiling": 0.30, "cer_converted": 0.35, "cer_excess_over_ceiling": 0.05,
         "ceiling_unusable": False, "asr_failed": False},
    ]

    def test_both_statistics_are_reported_with_distinct_names(self):
        from tools.asr_cer import summarise
        s = summarise(self.CLIPS)
        self.assertIn("cer_excess_diff_of_medians", s)
        self.assertIn("cer_excess_median_of_diffs", s)

    def test_the_two_statistics_differ_on_skewed_input(self):
        """**上限と変換の順位が clip ごとに違うと、2 つは大きく離れます。**

        上限 0.00 / 0.20 / 0.40（中央値 0.20）、変換 0.50 / 0.25 / 0.45（中央値 0.45）。
        **中央値の差は 0.25**、**差の中央値は 0.05** です。
        """
        from tools.asr_cer import summarise
        skew = [
            {"cer_ceiling": 0.00, "cer_converted": 0.50, "cer_excess_over_ceiling": 0.50,
             "ceiling_unusable": False, "asr_failed": False},
            {"cer_ceiling": 0.20, "cer_converted": 0.25, "cer_excess_over_ceiling": 0.05,
             "ceiling_unusable": False, "asr_failed": False},
            {"cer_ceiling": 0.40, "cer_converted": 0.45, "cer_excess_over_ceiling": 0.05,
             "ceiling_unusable": False, "asr_failed": False},
        ]
        s = summarise(skew)
        self.assertAlmostEqual(s["cer_excess_diff_of_medians"], 0.25, places=9)
        self.assertAlmostEqual(s["cer_excess_median_of_diffs"], 0.05, places=9)

    def test_the_old_name_still_means_the_old_thing(self):
        # 既存の記録と読み比べられるように、**旧い名前は旧い意味のまま**残す。
        from tools.asr_cer import summarise
        s = summarise(self.CLIPS)
        self.assertAlmostEqual(s["cer_excess_over_ceiling"],
                               s["cer_excess_diff_of_medians"], places=9)

    def test_degenerate_clips_are_excluded_from_the_median_of_diffs(self):
        # **変換 CER が 100% を超えた clip は ASR の暴走**。中央値に入れない
        # （`cer_breakdown.py` と同じ規則）。
        from tools.asr_cer import summarise
        bad = [*self.CLIPS, {"cer_ceiling": 0.0, "cer_converted": 9.0,
                             "cer_excess_over_ceiling": 9.0,
                             "ceiling_unusable": False, "asr_failed": False}]
        s = summarise(bad)
        self.assertEqual(s["n_degenerate"], 1)
        self.assertAlmostEqual(s["cer_excess_median_of_diffs"], 0.10, places=9)


class SpeakerSimilarityCollectionTests(unittest.TestCase):
    """どの WAV を「変換結果」として数えるか。**ここを間違えると別種のファイルを比べる。**"""

    def _dir(self, names):
        import tempfile
        from pathlib import Path
        d = tempfile.mkdtemp()
        for n in names:
            (Path(d) / n).write_bytes(b"")
        return Path(d)

    def test_ceiling_files_are_excluded(self):
        # `_vocoder_only` は **GT mel をボコーダーに通した上限**。変換結果として数えると
        # 自系だけが上限で嵩上げされ、baseline と別種のファイルを比べることになる。
        from tools.speaker_similarity import pick_clips
        d = self._dir(["a__c0_converted.wav", "a__c0_vocoder_only.wav",
                       "a__c0_source.wav", "b__c1_converted.wav"])
        got = sorted(p.name for p in pick_clips(d, n_clips=99))
        self.assertEqual(got, ["a__c0_converted.wav", "b__c1_converted.wav"])

    def test_plain_reference_directory_is_untouched(self):
        # target / unrelated は生の録音なので、名前で落とさないこと。
        from tools.speaker_similarity import pick_clips
        d = self._dir(["song_a.wav", "song_b.wav"])
        self.assertEqual(len(pick_clips(d, n_clips=99)), 2)

    def test_truncation_is_reported_not_silent(self):
        # **黙って 16 本に切ると、指標がフラグ次第で変わる。**
        from tools.speaker_similarity import pick_clips
        d = self._dir([f"s__c{i}_converted.wav" for i in range(20)])
        picked, avail = pick_clips(d, n_clips=16, with_count=True)
        self.assertEqual((len(picked), avail), (16, 20))

    def test_missing_directory_yields_nothing(self):
        from tools.speaker_similarity import pick_clips
        self.assertEqual(pick_clips(None, n_clips=16), [])


class TransposeResolutionTests(unittest.TestCase):
    """clip ごとの移調をどこから取るか。**黙って 0 と仮定しない。**"""

    def _clip(self, d, tag, transpose):
        import json
        from pathlib import Path
        p = Path(d) / f"song__{tag}_convert.json"
        p.write_text(json.dumps({"tag": tag, "transpose": transpose}), encoding="utf-8")

    def test_transpose_comes_from_the_clip_manifest(self):
        # 変換時に掛けた +12 を引かないと、**意図した移調が音程の誤りとして計上される**。
        import tempfile

        from tools.pitch_metrics import resolve_transposes
        with tempfile.TemporaryDirectory() as d:
            self._clip(d, "unseen00", 12)
            self._clip(d, "holdout00", 0)
            got = resolve_transposes(d)
            self.assertEqual(got["unseen00"], 12.0)
            self.assertEqual(got["holdout00"], 0.0)

    def test_testset_overrides_the_clip_manifest(self):
        import json
        import tempfile
        from pathlib import Path

        from tools.pitch_metrics import resolve_transposes
        with tempfile.TemporaryDirectory() as d:
            self._clip(d, "unseen00", 12)
            ts = Path(d) / "testset.json"
            ts.write_text(json.dumps({"unseen": [{"transpose": 7}]}), encoding="utf-8")
            self.assertEqual(resolve_transposes(d, testset=str(ts))["unseen00"], 7.0)

    def test_no_manifest_yields_no_entry_not_a_silent_zero(self):
        # 0 を返すと「移調なし」と区別が付かない。**入っていないこと自体を見せる。**
        import tempfile

        from tools.pitch_metrics import resolve_transposes
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(resolve_transposes(d), {})


class DefectPageTests(unittest.TestCase):
    """劣るほうの「何が悪いか」を名付けるページ。順位付けではなく名前を取る。"""

    ROWS = [{"pair": "pair02", "clip": "unseen17", "vote": "A"},
            {"pair": "pair09", "clip": "unseen14", "vote": "B"}]

    def test_it_asks_about_the_side_that_was_not_preferred(self):
        # 好んだほうの欠点を聞いても仕方がない。**負けたほうを聞く。**
        from tools.blind_test import defect_page
        html = defect_page(self.ROWS)
        self.assertIn('"ask": "B"', html.replace("'", '"'))
        self.assertIn('"ask": "A"', html.replace("'", '"'))

    def test_rows_without_a_vote_are_refused(self):
        # 票が無いと「劣るほう」が決まらない。
        from tools.blind_test import defect_page
        with self.assertRaises(ValueError):
            defect_page([{"pair": "pair02", "clip": "unseen17", "vote": ""}])

    def test_ties_are_refused(self):
        from tools.blind_test import defect_page
        with self.assertRaises(ValueError):
            defect_page([{"pair": "pair02", "clip": "unseen17", "vote": "tie"}])

    def test_system_names_never_reach_the_page(self):
        from tools.blind_test import defect_page
        html = defect_page(self.ROWS).lower()
        for name in ("leapsvc", "seedvc", "seed-vc"):
            self.assertNotIn(name, html)

    def test_key_rows_are_refused(self):
        from tools.blind_test import defect_page
        with self.assertRaises(ValueError):
            defect_page([{"pair": "pair02", "clip": "unseen17", "vote": "A",
                          "A": "leapsvc", "B": "seedvc"}])

    def test_header_has_no_comment_in_a_column_name(self):
        # `vote  # ...` で 26 票を失った。**同じ形を繰り返さない。**
        from tools.blind_test import DEFECT_HEADER
        for col in DEFECT_HEADER.split(","):
            self.assertNotIn("#", col)

    def test_the_label_set_covers_being_quieter_or_smaller(self):
        # 「音量が揺れる」しか無かったため、**「小さい」を「揺れる」と書かせてしまった**。
        # 選択肢に無い defect は、近いラベルに化けて記録される。
        from tools.blind_test import DEFECT_LABELS
        joined = "".join(DEFECT_LABELS)
        self.assertIn("小さい", joined)
        self.assertIn("遠い", joined)

    def test_every_defect_label_is_offered(self):
        from tools.blind_test import DEFECT_LABELS, defect_page
        html = defect_page(self.ROWS)
        for label in DEFECT_LABELS:
            self.assertIn(label, html)


class LoudnessStabilityTests(unittest.TestCase):
    """音量の揺れ。**blind で 6 本中 5 本が挙げた defect**を客観化する。"""

    def _tone(self, n, amp=0.2, sr=44100):
        import numpy as np
        t = np.arange(n) / sr
        return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    def test_constant_amplitude_gives_a_flat_envelope(self):
        import numpy as np

        from tools.loudness_stability import loudness_envelope
        env = loudness_envelope(self._tone(44100), sr=44100, hop=256)
        self.assertGreater(len(env), 100)
        self.assertLess(float(np.std(env)), 0.5)     # dB

    def test_a_gain_difference_alone_is_not_wobble(self):
        # 全体の音量差は揺れではない。**平均を取り除いてから見る。**
        from tools.loudness_stability import loudness_report
        src = self._tone(44100, amp=0.2)
        cnv = self._tone(44100, amp=0.05)            # 12 dB 小さいだけ
        r = loudness_report(src, cnv, sr=44100, hop=256)
        self.assertLess(r["residual_db_std"], 0.5)
        self.assertGreater(r["envelope_corr"], 0.5)

    def test_amplitude_modulation_raises_the_residual(self):
        import numpy as np

        from tools.loudness_stability import loudness_report
        n = 44100 * 2
        src = self._tone(n, amp=0.2)
        t = np.arange(n) / 44100
        wobble = (1.0 + 0.6 * np.sin(2 * np.pi * 4.0 * t)).astype(np.float32)
        r = loudness_report(src, src * wobble, sr=44100, hop=256)
        self.assertGreater(r["residual_db_std"], 2.0)

    def test_silence_does_not_dominate(self):
        # 無音区間の log-RMS は極端に小さい。**下限より下は使わない。**
        import numpy as np

        from tools.loudness_stability import loudness_report
        src = np.concatenate([self._tone(44100), np.zeros(44100, dtype="float32")])
        r = loudness_report(src, src.copy(), sr=44100, hop=256)
        self.assertLess(r["residual_db_std"], 0.5)
        # 2 秒ぶんのフレームがあるが、使うのは鳴っている 1 秒ぶんだけ。
        self.assertLess(r["n_frames_used"], r["n_frames_total"] * 0.7)

    def test_no_usable_frames_returns_none(self):
        import numpy as np

        from tools.loudness_stability import loudness_report
        z = np.zeros(44100, dtype="float32")
        r = loudness_report(z, z, sr=44100, hop=256)
        self.assertIsNone(r["residual_db_std"])

    def test_slow_drift_and_fast_wobble_are_separated(self):
        # 「揺れる」は**速さ**の話。ゆっくりした抑揚のずれと区別しないと捉えられない。
        import numpy as np

        from tools.loudness_stability import loudness_report
        sr, n = 44100, 44100 * 4
        t = np.arange(n) / sr
        base = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        slow = (1.0 + 0.5 * np.sin(2 * np.pi * 0.2 * t)).astype(np.float32)   # 0.2 Hz
        fast = (1.0 + 0.3 * np.sin(2 * np.pi * 6.0 * t)).astype(np.float32)   # 6 Hz
        rs = loudness_report(base, base * slow, sr=sr, hop=256)
        rf = loudness_report(base, base * fast, sr=sr, hop=256)
        self.assertGreater(rs["slow_db_std"], rs["fast_db_std"] * 3)
        self.assertGreater(rf["fast_db_std"], rf["slow_db_std"] * 3)

    def test_fast_component_is_none_without_usable_frames(self):
        import numpy as np

        from tools.loudness_stability import loudness_report
        z = np.zeros(44100, dtype="float32")
        self.assertIsNone(loudness_report(z, z, sr=44100, hop=256)["fast_db_std"])

    def _short_dips(self, x, *, sr, hop, every, width, gain, seed=0):
        """フレーム境界に**短い欠損**を置く。NHVSing V3.2 が直した defect の形。

        1 周期ぶん（数 ms）が打ち消されるので、**1 フレームより短い**落ち込みが
        飛び飛びに現れます。`fast_db_std`（2 Hz より速い成分すべて）はこれと
        数 Hz の抑揚のずれを区別しません。
        """
        import numpy as np
        rng = np.random.default_rng(seed)
        y = np.asarray(x, dtype=np.float32).copy()
        for f in range(0, len(y) // hop - 1, every):
            s = f * hop + int(rng.integers(0, hop))
            y[s:s + width] *= gain
        return y

    def test_frame_jitter_separates_per_frame_dips_from_a_6hz_wobble(self):
        # **これが V3.2 の defect を測るための量。** 6 Hz の抑揚のずれと、
        # フレーム単位の欠損は、`fast_db_std` ではどちらも「速い成分」になる。
        import numpy as np

        from tools.loudness_stability import loudness_report
        sr, hop, n = 44100, 256, 44100 * 4
        t = np.arange(n) / sr
        base = (0.2 * np.sin(2 * np.pi * 330 * t)).astype(np.float32)
        wobble = base * (1.0 + 0.3 * np.sin(2 * np.pi * 6.0 * t)).astype(np.float32)
        dips = self._short_dips(base, sr=sr, hop=hop, every=5, width=120, gain=0.15)

        rw = loudness_report(base, wobble, sr=sr, hop=hop)
        rd = loudness_report(base, dips, sr=sr, hop=hop)
        self.assertIn("frame_jitter_db", rw)
        # 従来の指標は 6 Hz の抑揚のずれのほうを「大きな揺れ」と見る（実測 1.88 対 0.52 dB）。
        self.assertGreater(rw["residual_db_std"], rd["residual_db_std"] * 2)
        self.assertGreater(rw["fast_db_std"], rd["fast_db_std"] * 2)
        # フレーム単位で見ると**順序が反転する**（実測 0.66 対 0.41 dB）。
        # **選択性は 1.6 倍しかありません** ―― 疎な欠損は std に薄まるので、
        # **この量だけで defect の有無を決めないこと。** 反転することだけが根拠になります。
        self.assertGreater(rd["frame_jitter_db"], rw["frame_jitter_db"])

    def test_frame_jitter_ignores_a_slow_drift(self):
        import numpy as np

        from tools.loudness_stability import loudness_report
        sr, n = 44100, 44100 * 4
        t = np.arange(n) / sr
        base = (0.2 * np.sin(2 * np.pi * 330 * t)).astype(np.float32)
        slow = base * (1.0 + 0.5 * np.sin(2 * np.pi * 0.2 * t)).astype(np.float32)
        r = loudness_report(base, slow, sr=sr, hop=256)
        self.assertIn("frame_jitter_db", r)
        self.assertGreater(r["residual_db_std"], 1.5)    # ゆっくり大きく動いている
        self.assertLess(r["frame_jitter_db"], 0.5)       # フレーム単位では静か

    def test_frame_jitter_is_none_without_usable_frames(self):
        import numpy as np

        from tools.loudness_stability import loudness_report
        z = np.zeros(44100, dtype="float32")
        r = loudness_report(z, z, sr=44100, hop=256)
        self.assertIn("frame_jitter_db", r)
        self.assertIsNone(r["frame_jitter_db"])

    def test_frame_jitter_does_not_cross_a_silent_gap(self):
        # 無音を挟んで飛ぶと、**そこだけ巨大な差**になる。隣り合う 2 フレームが
        # どちらも使える場合だけ差を取る。
        import numpy as np

        from tools.loudness_stability import loudness_report
        tone = self._tone(44100, amp=0.2)
        gap = np.zeros(22050, dtype="float32")
        src = np.concatenate([tone, gap, tone])
        r = loudness_report(src, src.copy(), sr=44100, hop=256)
        self.assertIn("frame_jitter_db", r)
        self.assertLess(r["frame_jitter_db"], 0.5)
