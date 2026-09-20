"""M0（データ確定）の道具のテスト。

    uv run python -m unittest test_svc_dataset -v

素材の品質検査と split 作成。実音声も重いモデルも使わない。
"""
from __future__ import annotations

import unittest

import numpy as np

from preprocess.svc.audit import AuditThresholds, audit_clip, effective_bandwidth_hz
from preprocess.svc.coverage import label_seconds, pitch_band_seconds, voiced_range
from preprocess.svc.report import build_report
from preprocess.svc.split import split_by_group


def _tone(n, sr=44100, hz=220.0, amp=0.5, seed=0):
    t = np.arange(n) / sr
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


class AuditClipTests(unittest.TestCase):
    """1 クリップを検査し、除外理由のリストを返す。

    実行計画 M0 ゴール 2「除外したものは reject reason つきで残してある」。
    合格なら空リスト。理由は人が読んで判断できる文字列にする。
    """

    SR = 44100

    def test_accepts_a_clean_clip(self):
        self.assertEqual(audit_clip(_tone(self.SR * 3), self.SR, expected_sr=self.SR), [])

    def test_rejects_a_sample_rate_mismatch(self):
        # mel 設定は前処理・loader・励起で共有されるので、sr の取り違えは静かに全部を壊す。
        reasons = audit_clip(_tone(22050 * 3, sr=22050), 22050, expected_sr=self.SR)
        self.assertTrue(any(r.startswith("sample_rate") for r in reasons), reasons)

    def test_rejects_a_clipped_clip(self):
        wav = _tone(self.SR, amp=1.5).clip(-1.0, 1.0)      # 振り切って潰れた波形
        reasons = audit_clip(wav, self.SR, expected_sr=self.SR)
        self.assertTrue(any(r.startswith("clipping") for r in reasons), reasons)

    def test_accepts_a_clip_that_merely_touches_full_scale(self):
        # 1 サンプルだけ 1.0 に触れるのは clipping ではない。閾値が緩すぎても厳しすぎても困る。
        wav = _tone(self.SR)
        wav[0] = 1.0
        self.assertEqual(audit_clip(wav, self.SR, expected_sr=self.SR), [])

    def test_rejects_a_mostly_silent_clip(self):
        wav = np.zeros(self.SR * 3, dtype=np.float32)
        wav[: self.SR // 10] = _tone(self.SR // 10)
        reasons = audit_clip(wav, self.SR, expected_sr=self.SR)
        self.assertTrue(any(r.startswith("silence") for r in reasons), reasons)

    def test_rejects_a_dc_offset(self):
        reasons = audit_clip(_tone(self.SR) + 0.2, self.SR, expected_sr=self.SR)
        self.assertTrue(any(r.startswith("dc_offset") for r in reasons), reasons)

    def test_rejects_a_clip_that_is_too_short(self):
        reasons = audit_clip(_tone(self.SR // 100), self.SR, expected_sr=self.SR)
        self.assertTrue(any(r.startswith("too_short") for r in reasons), reasons)

    def test_rejects_non_finite_samples(self):
        wav = _tone(self.SR).copy()
        wav[100] = np.nan
        reasons = audit_clip(wav, self.SR, expected_sr=self.SR)
        self.assertTrue(any(r.startswith("non_finite") for r in reasons), reasons)

    def test_reports_every_applicable_reason_not_just_the_first(self):
        # 1 つ直したら次が出る、では検査が何往復もする。まとめて返す。
        wav = np.zeros(22050 * 3, dtype=np.float32) + 0.9
        reasons = audit_clip(wav, 22050, expected_sr=self.SR)
        self.assertGreaterEqual(len(reasons), 2, reasons)

    def test_reasons_carry_the_measured_value(self):
        # 「clipping」だけでは閾値を調整できない。実測値を添える。
        wav = _tone(self.SR, amp=1.5).clip(-1.0, 1.0)
        reason = next(r for r in audit_clip(wav, self.SR, expected_sr=self.SR)
                      if r.startswith("clipping"))
        self.assertIn("=", reason)

    def test_thresholds_are_configurable(self):
        wav = _tone(self.SR)
        strict = AuditThresholds(max_dc_offset=0.0)
        self.assertNotEqual(audit_clip(wav, self.SR, expected_sr=self.SR, thresholds=strict), [])

    def test_rejects_a_non_mono_clip(self):
        with self.assertRaises(ValueError):
            audit_clip(np.zeros((2, 1000), dtype=np.float32), self.SR, expected_sr=self.SR)


class BandwidthTests(unittest.TestCase):
    """低い sample rate から上げただけの素材を見抜く。

    `mel.fmax` は 16,000 Hz。24 kHz 音源（Nyquist 12 kHz）を 44.1 kHz へ上げても
    12〜16 kHz は空のままで、混ぜて学習するとその帯域を「無い」と学習してこもった出力になる。
    JVS-MuSiC が配布版 24 kHz なのはまさにこの例で、GTSinger 自身も
    check_valid_bandwidth.py を同梱している。実在する問題なので検査する。
    """

    SR = 44100

    def _noise(self, n=44100, seed=0):
        return np.random.default_rng(seed).standard_normal(n).astype(np.float32) * 0.1

    def _band_limited(self, cutoff_hz, n=44100):
        # cutoff より上を落とした白色雑音（= その sr から上げた素材の模擬）
        rng = np.random.default_rng(1)
        spec = np.fft.rfft(rng.standard_normal(n))
        freqs = np.fft.rfftfreq(n, 1 / self.SR)
        spec[freqs > cutoff_hz] = 0.0
        y = np.fft.irfft(spec, n)
        return (y / (np.abs(y).max() + 1e-9) * 0.5).astype(np.float32)

    def test_full_band_noise_reaches_near_nyquist(self):
        bw = effective_bandwidth_hz(self._noise(), self.SR)
        self.assertGreater(bw, 0.9 * self.SR / 2)

    def test_detects_the_cutoff_of_band_limited_audio(self):
        bw = effective_bandwidth_hz(self._band_limited(12000.0), self.SR)
        self.assertAlmostEqual(bw, 12000.0, delta=800.0)

    def test_audit_flags_band_limited_audio_when_a_minimum_is_set(self):
        th = AuditThresholds(min_bandwidth_hz=16000.0)
        reasons = audit_clip(self._band_limited(12000.0), self.SR,
                             expected_sr=self.SR, thresholds=th)
        self.assertTrue(any(r.startswith("band_limited") for r in reasons), reasons)

    def test_audit_accepts_full_band_audio_when_a_minimum_is_set(self):
        th = AuditThresholds(min_bandwidth_hz=16000.0)
        reasons = audit_clip(self._noise(), self.SR, expected_sr=self.SR, thresholds=th)
        self.assertEqual([r for r in reasons if r.startswith("band_limited")], [])

    def test_bandwidth_check_is_off_by_default(self):
        # 純音のような合成信号は高域が無くて当然。既定で有効にすると誤検出だらけになる。
        reasons = audit_clip(self._band_limited(3000.0), self.SR, expected_sr=self.SR)
        self.assertEqual([r for r in reasons if r.startswith("band_limited")], [])


class SplitByGroupTests(unittest.TestCase):
    """曲・収録セッション単位で train/eval/test を分ける。

    実行計画 M0 ゴール 4。フレーズ単位で切ると同じ曲が train と test の両方に入り、
    leakage で性能を過大評価する。
    """

    def _names(self, n_groups=10, per_group=4):
        return {f"song{g:02d}_{i:04d}": f"song{g:02d}"
                for g in range(n_groups) for i in range(per_group)}

    def test_no_group_appears_in_two_splits(self):
        s = split_by_group(self._names(), seed=0, eval_groups=2, test_groups=2)
        for a, b in (("train", "eval"), ("train", "test"), ("eval", "test")):
            ga = {self._names()[n] for n in s[a]}
            gb = {self._names()[n] for n in s[b]}
            self.assertEqual(ga & gb, set(), f"{a} と {b} に同じ group がある")

    def test_every_name_lands_in_exactly_one_split(self):
        names = self._names()
        s = split_by_group(names, seed=0, eval_groups=2, test_groups=2)
        allocated = s["train"] + s["eval"] + s["test"]
        self.assertEqual(sorted(allocated), sorted(names))
        self.assertEqual(len(allocated), len(set(allocated)))

    def test_requested_group_counts_are_honoured(self):
        names = self._names()
        s = split_by_group(names, seed=0, eval_groups=3, test_groups=2)
        self.assertEqual(len({names[n] for n in s["eval"]}), 3)
        self.assertEqual(len({names[n] for n in s["test"]}), 2)

    def test_same_seed_gives_the_same_split(self):
        names = self._names()
        a = split_by_group(names, seed=7, eval_groups=2, test_groups=2)
        b = split_by_group(names, seed=7, eval_groups=2, test_groups=2)
        self.assertEqual(a, b)

    def test_different_seed_gives_a_different_split(self):
        names = self._names(n_groups=20)
        a = split_by_group(names, seed=0, eval_groups=3, test_groups=3)
        b = split_by_group(names, seed=1, eval_groups=3, test_groups=3)
        self.assertNotEqual(a, b)

    def test_train_keeps_at_least_one_group(self):
        with self.assertRaises(ValueError):
            split_by_group(self._names(n_groups=4), seed=0, eval_groups=2, test_groups=2)

    def test_rejects_an_empty_dataset(self):
        with self.assertRaises(ValueError):
            split_by_group({}, seed=0, eval_groups=1, test_groups=1)

    def test_names_within_a_split_are_sorted(self):
        # split list はファイルに書いて差分を読むもの。順序が安定しないと差分が意味を失う。
        s = split_by_group(self._names(), seed=0, eval_groups=2, test_groups=2)
        for k in ("train", "eval", "test"):
            self.assertEqual(s[k], sorted(s[k]))


class CoverageTests(unittest.TestCase):
    """音域の滞在時間を集計する（実行計画 M0 ゴール 3）。

    高音・裏声が薄い素材で学習すると、そこだけ崩れる。学習前に分布を見るための集計。
    """

    FR = 44100 / 256   # 172.265625 Hz

    def test_counts_only_voiced_frames(self):
        f0 = np.array([220.0, 220.0, 220.0, 220.0], dtype=np.float32)
        uv = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
        total = sum(pitch_band_seconds(f0, uv, frame_rate=self.FR, edges_hz=[300.0]).values())
        self.assertAlmostEqual(total, 2 / self.FR, places=6)

    def test_assigns_frames_to_bands_by_the_given_edges(self):
        f0 = np.array([100.0, 250.0, 600.0], dtype=np.float32)
        uv = np.ones(3, dtype=np.float32)
        bands = pitch_band_seconds(f0, uv, frame_rate=self.FR, edges_hz=[200.0, 400.0])
        self.assertEqual(len(bands), 3)
        for seconds in bands.values():
            self.assertAlmostEqual(seconds, 1 / self.FR, places=6)

    def test_a_value_on_an_edge_goes_to_the_upper_band(self):
        f0 = np.array([200.0], dtype=np.float32)
        uv = np.ones(1, dtype=np.float32)
        bands = pitch_band_seconds(f0, uv, frame_rate=self.FR, edges_hz=[200.0])
        labels = list(bands)
        self.assertAlmostEqual(bands[labels[0]], 0.0, places=9)
        self.assertAlmostEqual(bands[labels[1]], 1 / self.FR, places=6)

    def test_all_unvoiced_gives_zero_everywhere_without_raising(self):
        f0 = np.array([220.0, 330.0], dtype=np.float32)
        uv = np.zeros(2, dtype=np.float32)
        bands = pitch_band_seconds(f0, uv, frame_rate=self.FR, edges_hz=[300.0])
        self.assertEqual(set(bands.values()), {0.0})

    def test_rejects_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            pitch_band_seconds(np.zeros(3, dtype=np.float32), np.zeros(2, dtype=np.float32),
                               frame_rate=self.FR, edges_hz=[300.0])

    def test_rejects_unsorted_edges(self):
        with self.assertRaises(ValueError):
            pitch_band_seconds(np.array([220.0], dtype=np.float32),
                               np.ones(1, dtype=np.float32),
                               frame_rate=self.FR, edges_hz=[400.0, 200.0])

    def test_voiced_range_reports_percentiles_in_hz(self):
        f0 = np.linspace(100.0, 500.0, 101).astype(np.float32)
        uv = np.ones(101, dtype=np.float32)
        r = voiced_range(f0, uv, frame_rate=self.FR)
        self.assertAlmostEqual(r["p50_hz"], 300.0, delta=1.0)
        self.assertLess(r["p05_hz"], r["p50_hz"])
        self.assertGreater(r["p95_hz"], r["p50_hz"])

    def test_voiced_range_reports_the_span_in_semitones(self):
        # 1 オクターブ = 12 半音。音域の広さは Hz より半音のほうが読める。
        f0 = np.array([220.0, 440.0], dtype=np.float32)
        uv = np.ones(2, dtype=np.float32)
        r = voiced_range(f0, uv, frame_rate=self.FR)
        self.assertAlmostEqual(r["span_semitones"], 12.0, places=3)

    def test_voiced_range_reports_voiced_seconds(self):
        f0 = np.full(10, 220.0, dtype=np.float32)
        uv = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        r = voiced_range(f0, uv, frame_rate=self.FR)
        self.assertAlmostEqual(r["voiced_sec"], 3 / self.FR, places=6)

    def test_voiced_range_survives_a_fully_unvoiced_clip(self):
        r = voiced_range(np.zeros(4, dtype=np.float32), np.zeros(4, dtype=np.float32),
                         frame_rate=self.FR)
        self.assertEqual(r["voiced_sec"], 0.0)
        self.assertTrue(all(v == 0.0 or np.isnan(v) is False for v in r.values()))


class StratifiedSplitTests(unittest.TestCase):
    """held-out に偏りが出ないよう層で均す。

    実データ（VocalSet 20 名）で確認した問題: 歌手単位でランダムに held-out を選ぶと、
    seed 0 では eval も test も全員男性、seed 3 では全員女性になった。SVC は異性間の変換が
    難しい方なので、held-out が片方の性別だけだと評価がその難所を素通りする。
    """

    def _names(self, n_each=8):
        names, strata = {}, {}
        for i in range(n_each):
            for sex in ("f", "m"):
                g = f"{sex}{i}"
                strata[g] = sex
                for k in range(2):
                    names[f"{g}_{k:04d}"] = g
        return names, strata

    def test_each_split_gets_a_mix_of_strata(self):
        names, strata = self._names()
        for seed in range(5):
            with self.subTest(seed=seed):
                s = split_by_group(names, seed=seed, eval_groups=2, test_groups=2,
                                   strata=strata)
                for key in ("eval", "test"):
                    got = {strata[names[n]] for n in s[key]}
                    self.assertEqual(got, {"f", "m"}, f"{key} が片方の層だけ (seed={seed})")

    def test_still_honours_the_requested_group_counts(self):
        names, strata = self._names()
        s = split_by_group(names, seed=0, eval_groups=2, test_groups=2, strata=strata)
        self.assertEqual(len({names[n] for n in s["eval"]}), 2)
        self.assertEqual(len({names[n] for n in s["test"]}), 2)

    def test_no_group_appears_in_two_splits(self):
        names, strata = self._names()
        s = split_by_group(names, seed=1, eval_groups=2, test_groups=2, strata=strata)
        for a, b in (("train", "eval"), ("train", "test"), ("eval", "test")):
            ga = {names[n] for n in s[a]}
            gb = {names[n] for n in s[b]}
            self.assertEqual(ga & gb, set())

    def test_is_deterministic(self):
        names, strata = self._names()
        a = split_by_group(names, seed=2, eval_groups=2, test_groups=2, strata=strata)
        b = split_by_group(names, seed=2, eval_groups=2, test_groups=2, strata=strata)
        self.assertEqual(a, b)

    def test_falls_back_when_a_stratum_runs_out(self):
        # 層の数より held-out が多いときも、要求した group 数は満たす。
        names, strata = self._names(n_each=4)
        s = split_by_group(names, seed=0, eval_groups=3, test_groups=1, strata=strata)
        self.assertEqual(len({names[n] for n in s["eval"]}), 3)

    def test_rejects_a_stratum_map_missing_a_group(self):
        names, strata = self._names()
        del strata["f0"]
        with self.assertRaises(ValueError):
            split_by_group(names, seed=0, eval_groups=2, test_groups=2, strata=strata)


class LabelSecondsTests(unittest.TestCase):
    """ラベルごとの滞在秒数（実行計画 M0 ゴール 3 の「発声スタイルの coverage」）。

    技法ラベルを持つ corpus（GTSinger / VocalSet）なら、これで発声スタイルの偏りが分かる。
    区間の長さで重み付けする。区間数を数えても、短い区間が多いだけで多いことになってしまう。
    """

    def test_sums_durations_per_label(self):
        out = label_seconds(["belt", "breathy", "belt"], [1.0, 2.0, 0.5])
        self.assertAlmostEqual(out["belt"], 1.5)
        self.assertAlmostEqual(out["breathy"], 2.0)

    def test_orders_by_descending_seconds(self):
        # 偏りを見るための集計なので、多い順に並んでいないと読めない。
        out = label_seconds(["a", "b", "c"], [1.0, 3.0, 2.0])
        self.assertEqual(list(out), ["b", "c", "a"])

    def test_breaks_ties_by_label_for_determinism(self):
        out = label_seconds(["z", "a"], [1.0, 1.0])
        self.assertEqual(list(out), ["a", "z"])

    def test_returns_empty_for_no_segments(self):
        self.assertEqual(label_seconds([], []), {})

    def test_rejects_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            label_seconds(["a", "b"], [1.0])

    def test_rejects_negative_durations(self):
        with self.assertRaises(ValueError):
            label_seconds(["a"], [-1.0])


class BuildReportTests(unittest.TestCase):
    """M0 の成果物（reject list / coverage / split list / manifest）を 1 度に作る。

    実行計画 M0 の成果物は「dataset ledger、split list、reject list、coverage 集計」。
    道具が関数として在るだけでは、素材が届いたときに毎回つなぎを書くことになる。
    音声の読み込みは差し替え可能にして、重いモデルもファイル I/O も使わずにテストする。
    """

    SR = 44100

    def _loader(self, bad: set[str]):
        """name -> (wav, sr) を返す偽のローダー。bad に入れた名前は無音（= 弾かれる）。"""
        def load(name):
            if name in bad:
                return np.zeros(self.SR * 2, dtype=np.float32), self.SR
            return _tone(self.SR * 2, sr=self.SR), self.SR
        return load

    def _names(self, n_groups=6, per_group=3):
        return {f"song{g:02d}_{i:04d}": f"song{g:02d}"
                for g in range(n_groups) for i in range(per_group)}

    def test_records_every_rejected_clip_with_its_reasons(self):
        names = self._names()
        bad = {"song00_0000", "song03_0001"}
        r = build_report(names, self._loader(bad), expected_sr=self.SR,
                         seed=0, eval_groups=1, test_groups=1)
        self.assertEqual(sorted(r["rejects"]), sorted(bad))
        for name in bad:
            self.assertTrue(r["rejects"][name], f"{name} に理由が付いていない")

    def test_rejected_clips_never_appear_in_any_split(self):
        # 弾いた素材が split に残っていると、学習が起動時に落ちる。
        names = self._names()
        bad = {"song00_0000"}
        r = build_report(names, self._loader(bad), expected_sr=self.SR,
                         seed=0, eval_groups=1, test_groups=1)
        for key in ("train", "eval", "test"):
            self.assertNotIn("song00_0000", r["split"][key])

    def test_accepted_clips_are_all_placed(self):
        names = self._names()
        bad = {"song00_0000"}
        r = build_report(names, self._loader(bad), expected_sr=self.SR,
                         seed=0, eval_groups=1, test_groups=1)
        placed = r["split"]["train"] + r["split"]["eval"] + r["split"]["test"]
        self.assertEqual(sorted(placed), sorted(set(names) - bad))

    def test_manifest_records_what_is_needed_to_reproduce(self):
        r = build_report(self._names(), self._loader(set()), expected_sr=self.SR,
                         seed=3, eval_groups=1, test_groups=1)
        m = r["manifest"]
        for key in ("seed", "eval_groups", "test_groups", "expected_sr", "thresholds"):
            self.assertIn(key, m)
        self.assertEqual(m["seed"], 3)

    def test_reports_accepted_and_rejected_durations(self):
        names = self._names(n_groups=2, per_group=2)
        r = build_report(names, self._loader({"song00_0000"}), expected_sr=self.SR,
                         seed=0, eval_groups=1, test_groups=0)
        self.assertAlmostEqual(r["totals"]["accepted_sec"], 3 * 2.0, places=3)
        self.assertAlmostEqual(r["totals"]["rejected_sec"], 1 * 2.0, places=3)

    def test_includes_label_coverage_when_labels_are_given(self):
        # 技法や性別のラベルがあるなら、reject / split と同じ一度の走査で集計まで出す。
        names = self._names(n_groups=4, per_group=2)
        labels = {n: ("belt" if n.startswith("song00") else "breathy") for n in names}
        r = build_report(names, self._loader(set()), expected_sr=self.SR,
                         seed=0, eval_groups=1, test_groups=1, labels=labels)
        cov = r["coverage"]["labels"]
        self.assertAlmostEqual(cov["belt"], 2 * 2.0, places=3)
        self.assertAlmostEqual(cov["breathy"], 6 * 2.0, places=3)

    def test_label_coverage_counts_only_accepted_clips(self):
        # 弾いた素材を coverage に混ぜると、実際に学習する分布と食い違う。
        names = self._names(n_groups=4, per_group=2)
        labels = {n: "belt" for n in names}
        r = build_report(names, self._loader({"song00_0000"}), expected_sr=self.SR,
                         seed=0, eval_groups=1, test_groups=1, labels=labels)
        self.assertAlmostEqual(r["coverage"]["labels"]["belt"], 7 * 2.0, places=3)

    def test_coverage_is_absent_without_labels(self):
        r = build_report(self._names(), self._loader(set()), expected_sr=self.SR,
                         seed=0, eval_groups=1, test_groups=1)
        self.assertNotIn("labels", r.get("coverage", {}))

    def test_is_deterministic_for_the_same_seed(self):
        names = self._names()
        a = build_report(names, self._loader(set()), expected_sr=self.SR,
                         seed=5, eval_groups=1, test_groups=1)
        b = build_report(names, self._loader(set()), expected_sr=self.SR,
                         seed=5, eval_groups=1, test_groups=1)
        self.assertEqual(a["split"], b["split"])

    def test_raises_when_every_clip_is_rejected(self):
        # 全滅は「検査が厳しすぎる」か「素材が壊れている」。黙って空の split を返さない。
        names = self._names(n_groups=3)
        with self.assertRaises(ValueError):
            build_report(names, self._loader(set(names)), expected_sr=self.SR,
                         seed=0, eval_groups=1, test_groups=1)

    def test_a_group_losing_every_clip_drops_out_of_the_split(self):
        names = self._names(n_groups=4, per_group=2)
        bad = {"song00_0000", "song00_0001"}
        r = build_report(names, self._loader(bad), expected_sr=self.SR,
                         seed=0, eval_groups=1, test_groups=1)
        placed = r["split"]["train"] + r["split"]["eval"] + r["split"]["test"]
        self.assertEqual([n for n in placed if n.startswith("song00")], [])


class JaMaterialAuditTests(unittest.TestCase):
    """日本語素材の棚卸し。**「学習に入っていない曲」を曲単位で出す。**

    base の 23 話者には**日本語 5 名が全員入っています**（`JA_Soprano_1` / `JA_Tenor_1` /
    `natsume` / `oniku` / `ritsu`）。未知話者は作れないので、作れるのは **「未知曲」**
    までです。**曲単位で漏れなく判定しないと leakage します。**

    GTSinger のパスは `<技法>/<曲>/<Group>/NNNN.wav` で、**同じ曲が複数の技法の下に
    出ます**。どれか 1 つでも学習に入っていれば、その曲は使えません。
    """

    def test_song_comes_from_the_second_path_component(self):
        from tools.ja_material_audit import song_of
        self.assertEqual(song_of("Breathy/Heartful Song/Breathy_Group/0000.wav"),
                         "heartful song")

    def test_song_names_are_folded(self):
        # **既知の落とし穴**: `Heartful_Song` と `Heartful_song` が別の曲になると split が効かない。
        from tools.ja_material_audit import song_of
        a = song_of("Breathy/Heartful_Song/G/0.wav")
        b = song_of("Glissando/Heartful song/G/1.wav")
        self.assertEqual(a, b)

    def test_flat_layouts_take_the_song_from_the_file(self):
        # UTAU DB は **1 ファイル = 1 曲**（`wav/1.wav` / `akai_kutsu/akai_kutsu.wav` /
        # `DATABASE/<曲>/<曲>.wav`）。GTSinger の規則を当てると曲を取り違えます。
        from tools.ja_material_audit import song_of
        self.assertEqual(song_of("wav/1.wav", layout="flat"), "1")
        self.assertEqual(song_of("akai_kutsu/akai_kutsu.wav", layout="flat"), "akai kutsu")
        self.assertEqual(song_of("DATABASE/ARROW+3_normal/ARROW+3_normal.wav",
                                 layout="flat"), "arrow+3 normal")

    def test_the_layout_must_be_known(self):
        # **黙って既定に落とすと曲を取り違えて leakage します。**
        from tools.ja_material_audit import song_of
        with self.assertRaises(ValueError):
            song_of("wav/1.wav", layout="なんとなく")

    def test_audit_passes_the_layout_through(self):
        from tools.ja_material_audit import audit
        rep = audit(used=["wav/1.wav"], disk=["wav/1.wav", "wav/2.wav"], layout="flat")
        self.assertEqual(rep["used_songs"], ["1"])
        self.assertEqual(rep["free_songs"], ["2"])

    def test_a_song_used_under_any_technique_is_contaminated(self):
        # 1 つの技法でも学習に入っていれば、その曲は test に使えない。
        from tools.ja_material_audit import audit
        rep = audit(used=["Breathy/S1/G/0000.wav"],
                    disk=["Breathy/S1/G/0000.wav", "Glissando/S1/G/0000.wav",
                          "Breathy/S2/G/0000.wav"])
        self.assertEqual(rep["used_songs"], ["s1"])
        self.assertEqual(rep["free_songs"], ["s2"])

    def test_files_are_matched_by_relative_path_not_by_name(self):
        # **連番は曲ごとに振り直されます**（どの曲にも 0000.wav がある）。名前で照合すると
        # 全部が「学習済み」に見えます。実際にこれで数を間違えました。
        from tools.ja_material_audit import audit
        rep = audit(used=["Breathy/S1/G/0000.wav"],
                    disk=["Breathy/S1/G/0000.wav", "Breathy/S2/G/0000.wav"])
        self.assertEqual(rep["free_songs"], ["s2"])
        self.assertEqual(rep["n_free_files"], 1)

    def test_files_in_the_manifest_but_not_on_disk_are_reported(self):
        # **黙って無視すると、学習素材が欠けていることに気づけません。**
        from tools.ja_material_audit import audit
        rep = audit(used=["Breathy/S1/G/0000.wav", "Breathy/S1/G/0001.wav"],
                    disk=["Breathy/S1/G/0000.wav"])
        self.assertEqual(rep["missing_from_disk"], ["Breathy/S1/G/0001.wav"])

    def test_hash_mismatches_are_reported(self):
        # 手元の素材が学習に使ったものと同じであることを確かめる。
        from tools.ja_material_audit import audit
        rep = audit(used={"Breathy/S1/G/0000.wav": "aa"},
                    disk=["Breathy/S1/G/0000.wav", "Breathy/S2/G/0000.wav"],
                    disk_hashes={"Breathy/S1/G/0000.wav": "bb"})
        self.assertEqual(rep["hash_mismatch"], ["Breathy/S1/G/0000.wav"])

    def test_matching_hashes_are_not_reported(self):
        from tools.ja_material_audit import audit
        rep = audit(used={"Breathy/S1/G/0000.wav": "aa"},
                    disk=["Breathy/S1/G/0000.wav"],
                    disk_hashes={"Breathy/S1/G/0000.wav": "aa"})
        self.assertEqual(rep["hash_mismatch"], [])


class JaTestsetTests(unittest.TestCase):
    """日本語 test set の構築。**学習に入った曲を絶対に混ぜないことが最優先の契約。**

    手元の日本語 5 名は**全員 base に入っています**。作れるのは「**未知曲・既知話者**」
    までで、**未知話者とは名乗れません**。曲の選別を間違えると、その区別すら失われます。
    """

    AUDIT = {"natsume": {"root": "download/natsume",
                         "used_songs": ["1", "2"],
                         "free_songs": ["36", "37"],
                         "free_files": ["wav/36.wav", "wav/37.wav"]}}

    def test_the_pool_only_contains_unused_songs(self):
        from tools.ja_testset import pool_from_audit
        pool = pool_from_audit(self.AUDIT, speaker="natsume", layout="flat",
                               seconds=20.0, durations={"wav/36.wav": 60.0,
                                                        "wav/37.wav": 60.0}, seen_speaker=True)
        self.assertEqual(sorted(p["song"] for p in pool), ["36", "37"])

    def test_a_used_song_can_never_enter_the_pool(self):
        # free_files に学習済みの曲が紛れていたら落とす。**黙って通すと leakage します。**
        from tools.ja_testset import pool_from_audit
        bad = {"natsume": {**self.AUDIT["natsume"],
                           "free_files": ["wav/1.wav", "wav/36.wav"]}}
        with self.assertRaises(ValueError):
            pool_from_audit(bad, speaker="natsume", layout="flat", seconds=20.0,
                            durations={"wav/1.wav": 60.0, "wav/36.wav": 60.0}, seen_speaker=True)

    def test_songs_shorter_than_the_clip_are_dropped(self):
        from tools.ja_testset import pool_from_audit
        pool = pool_from_audit(self.AUDIT, speaker="natsume", layout="flat",
                               seconds=20.0, durations={"wav/36.wav": 10.0,
                                                        "wav/37.wav": 60.0}, seen_speaker=True)
        self.assertEqual([p["song"] for p in pool], ["37"])

    def test_clips_below_the_similarity_minimum_are_refused(self):
        # `speaker_similarity.py` は 12 秒未満のクリップを拒否します。
        from tools.ja_testset import pool_from_audit
        with self.assertRaises(ValueError):
            pool_from_audit(self.AUDIT, speaker="natsume", layout="flat", seconds=6.0,
                            durations={"wav/36.wav": 60.0, "wav/37.wav": 60.0}, seen_speaker=True)

    def test_the_pool_records_what_it_is(self):
        # **「未知話者」と名乗らない。** 層を記録に残す。
        from tools.ja_testset import pool_from_audit
        pool = pool_from_audit(self.AUDIT, speaker="natsume", layout="flat",
                               seconds=20.0, durations={"wav/36.wav": 60.0}, seen_speaker=True)
        self.assertEqual(pool[0]["speaker"], "natsume")
        self.assertEqual(pool[0]["kind"], "unseen_song")
        self.assertIn("seen_speaker", pool[0])
        self.assertTrue(pool[0]["seen_speaker"])

    def test_the_tier_is_not_hardcoded(self):
        """**層を決め打ちしないこと。**

        `natsume` / `oniku` は base に入っているので「未知曲・既知話者」ですが、
        **東北きりたん / No.7 は入っていない**ので「未知話者」です。決め打ちすると
        **一番言いたい主張がラベルとして間違って残ります**（実際に踏みました）。
        """
        from tools.ja_testset import pool_from_audit
        seen = pool_from_audit(self.AUDIT, speaker="natsume", layout="flat", seconds=20.0,
                               durations={"wav/36.wav": 60.0}, seen_speaker=True)
        self.assertTrue(seen[0]["seen_speaker"])
        self.assertEqual(seen[0]["kind"], "unseen_song")
        unseen = pool_from_audit(self.AUDIT, speaker="natsume", layout="flat", seconds=20.0,
                                 durations={"wav/36.wav": 60.0}, seen_speaker=False)
        self.assertFalse(unseen[0]["seen_speaker"])
        self.assertEqual(unseen[0]["kind"], "unseen_speaker")

    def test_an_unknown_speaker_is_refused(self):
        from tools.ja_testset import pool_from_audit
        with self.assertRaises(ValueError):
            pool_from_audit(self.AUDIT, speaker="kiritan", layout="flat", seconds=20.0,
                            durations={}, seen_speaker=True)


class HoldoutReproductionTests(unittest.TestCase):
    r"""**学習 hold-out を後から再現できること。**

    `SVCFeatureDataset` の split は `random.Random(seed).sample(sorted(曲名), n)` です。
    曲名は `preprocess.svc.run` が **casefold してから** `[^\w-]+` を `_` に置き換えた形で、
    **casefold を忘れると並び順が変わり、別の曲が hold-out になります**（実際に一度
    取り違えました）。

    再現できると、**どの曲が学習に入っていたか**を後から判定できます。これは
    「明瞭度の劣化が容量の問題か汎化の問題か」を切り分けるのに要ります。
    """

    def test_casefold_changes_which_songs_are_held_out(self):
        import random
        raw = ["ARROW+3_normal", "anywhere-3_normal", "Baptism+3_normal",
               "boukyakumoyou-3_normal", "skyhighblue-3_normal", "silentrail_normal"]
        import re
        safe = re.compile(r"[^\w-]+")
        plain = sorted({safe.sub("_", s) for s in raw})
        folded = sorted({safe.sub("_", s.casefold()) for s in raw})
        self.assertNotEqual(random.Random(42).sample(plain, 3),
                            random.Random(42).sample(folded, 3))

    def test_the_split_is_reproducible_from_the_song_list(self):
        # 同じ曲名リストと seed からは必ず同じ hold-out が出る（後から判定できる根拠）。
        import random
        songs = sorted(f"song{i:02d}" for i in range(50))
        a = sorted(random.Random(42).sample(songs, 3))
        b = sorted(random.Random(42).sample(songs, 3))
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()


class GTSingerPickWavsTests(unittest.TestCase):
    """M3 の素材取得: 歌手ごとに wav を等間隔で選ぶ。

    GTSinger のリポジトリは 149,037 ファイルあり、丸ごと落とすと HTTP 429 で律速されて
    3 時間半かかります（実測）。使う wav だけを選んで落とします。**先頭から取ってはいけません**
    — パスは技法名でソートされるので `Breathy` ばかりになります。
    """

    def _files(self):
        out = ["README.md", "Japanese/JA-Soprano-1/Breathy/song/G/0000.json"]
        for singer in ("JA-Soprano-1", "JA-Tenor-1"):
            for tech in ("Breathy", "Vibrato", "Glissando"):
                for i in range(10):
                    out.append(f"Japanese/{singer}/{tech}/song{i//5}/G/{i:04d}.wav")
        out += [f"Chinese/ZH-Alto-1/Breathy/s/G/{i:04d}.wav" for i in range(5)]
        return out

    def test_takes_only_wavs_of_the_requested_languages(self):
        from tools.m3_corpus import pick_wavs
        picked = pick_wavs(self._files(), ["Japanese"], 0)
        self.assertTrue(all(p.endswith(".wav") for p in picked))
        self.assertTrue(all(p.startswith("Japanese/") for p in picked))

    def test_caps_each_singer_independently(self):
        from tools.m3_corpus import pick_wavs
        picked = pick_wavs(self._files(), ["Japanese"], 6)
        per = {}
        for p in picked:
            per[p.split("/")[1]] = per.get(p.split("/")[1], 0) + 1
        self.assertEqual(per, {"JA-Soprano-1": 6, "JA-Tenor-1": 6})

    def test_spreads_across_techniques_instead_of_taking_the_first_ones(self):
        from tools.m3_corpus import pick_wavs
        picked = [p for p in pick_wavs(self._files(), ["Japanese"], 6)
                  if "JA-Soprano-1" in p]
        self.assertEqual({p.split("/")[2] for p in picked},
                         {"Breathy", "Vibrato", "Glissando"}, "技法が偏っている")

    def test_returns_everything_when_no_cap_is_given(self):
        from tools.m3_corpus import pick_wavs
        self.assertEqual(len(pick_wavs(self._files(), ["Japanese", "Chinese"], 0)), 65)

    def test_is_deterministic(self):
        from tools.m3_corpus import pick_wavs
        self.assertEqual(pick_wavs(self._files(), ["Japanese"], 7),
                         pick_wavs(self._files(), ["Japanese"], 7))


class NhvIndistExtraSetTests(unittest.TestCase):
    """`tools/nhv_indist.py` に任意のコーパスを足す（doc/svc-plan.md 12 節 P0-3）。

    既存 5 セットはモジュール定数で、**新しい素材を並べる口がありませんでした**。
    7b 節と同一条件で比べるために、セットだけを足せるようにします。
    """

    def _parse(self, values):
        import importlib.util
        from pathlib import Path
        spec = importlib.util.spec_from_file_location(
            "nhv_indist", Path(__file__).resolve().parent / "tools" / "nhv_indist.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.parse_extra_sets(values)

    def test_name_and_path_are_parsed(self):
        sets = self._parse(["tts=.m0data/p0/vocal"])
        self.assertEqual(len(sets), 1)
        name, rel, in_nhv, note = sets[0]
        self.assertEqual(name, "tts")
        self.assertEqual(rel, ".m0data/p0/vocal")

    def test_an_extra_set_is_never_marked_as_nhvsing_training_data(self):
        """既知/未知の列は**素材の事実**なので、足した側が勝手に「既知」を名乗らない。"""
        _, _, in_nhv, _ = self._parse(["tts=.m0data/p0/vocal"])[0]
        self.assertFalse(in_nhv)

    def test_a_note_can_be_given(self):
        _, _, _, note = self._parse(["tts=.m0data/p0/vocal=分離ボーカル"])[0]
        self.assertEqual(note, "分離ボーカル")

    def test_several_sets_are_parsed(self):
        self.assertEqual(len(self._parse(["a=x", "b=y"])), 2)

    def test_a_value_without_an_equals_sign_is_rejected(self):
        with self.assertRaises(ValueError):
            self._parse([".m0data/p0/vocal"])

    def test_an_empty_name_is_rejected(self):
        with self.assertRaises(ValueError):
            self._parse(["=.m0data/p0/vocal"])

    def test_a_name_colliding_with_a_default_set_is_rejected(self):
        """既存セットを黙って置き換えると、7b 節の値と比較できなくなる。"""
        with self.assertRaises(ValueError):
            self._parse(["ritsu=.m0data/p0/vocal"])

    def test_none_gives_no_sets(self):
        self.assertEqual(self._parse(None), [])


class P1CorpusSelectionTests(unittest.TestCase):
    """1,000 h 素材から base 用の話者を選ぶ（doc/svc-plan.md 13 節 P1）。

    **話者を稼ぐのが目的**なので、同じ channel から何曲も取らない。`official` は
    channel = 話者にならない（`Warner Music Japan` が実在）ので除く。
    """

    def _entries(self, n=10, cat="cover", sec=240):
        return [{"video_id": f"v{i}", "channel_id": f"c{i}",
                 "dataset_category": cat, "duration_sec": sec} for i in range(n)]

    def _plan(self, entries, **kw):
        import importlib.util
        from pathlib import Path
        spec = importlib.util.spec_from_file_location(
            "p1_corpus", Path(__file__).resolve().parent / "tools" / "p1_corpus.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.plan_selection(entries, **kw)

    def test_only_cover_is_selected(self):
        mixed = self._entries(3, "cover") + self._entries(3, "official") + self._entries(3, "diverse")
        got = self._plan(mixed, hours=10.0, seed=0)
        self.assertTrue(all(e["dataset_category"] == "cover" for e in got))

    def test_an_entry_without_a_channel_is_dropped(self):
        entries = self._entries(2)
        entries[0]["channel_id"] = ""
        got = self._plan(entries, hours=10.0, seed=0)
        self.assertEqual([e["video_id"] for e in got], ["v1"])

    def test_the_hour_budget_stops_the_selection(self):
        got = self._plan(self._entries(100, sec=360), hours=1.0, seed=0)
        self.assertEqual(len(got), 10)          # 360 s x 10 = 1.0 h

    def test_one_song_per_speaker_by_default(self):
        entries = [{"video_id": f"v{i}", "channel_id": "same",
                    "dataset_category": "cover", "duration_sec": 240} for i in range(5)]
        got = self._plan(entries, hours=10.0, seed=0)
        self.assertEqual(len(got), 1)

    def test_per_speaker_cap_can_be_raised(self):
        entries = [{"video_id": f"v{i}", "channel_id": "same",
                    "dataset_category": "cover", "duration_sec": 240} for i in range(5)]
        got = self._plan(entries, hours=10.0, seed=0, per_speaker=3)
        self.assertEqual(len(got), 3)

    def test_the_same_seed_reproduces_the_selection(self):
        entries = self._entries(50)
        a = self._plan(entries, hours=1.0, seed=7)
        b = self._plan(entries, hours=1.0, seed=7)
        self.assertEqual([e["video_id"] for e in a], [e["video_id"] for e in b])

    def test_a_different_seed_changes_the_selection(self):
        entries = self._entries(50)
        a = self._plan(entries, hours=1.0, seed=0)
        b = self._plan(entries, hours=1.0, seed=1)
        self.assertNotEqual([e["video_id"] for e in a], [e["video_id"] for e in b])

    def test_entries_without_a_duration_are_dropped(self):
        entries = self._entries(2)
        entries[0].pop("duration_sec")
        got = self._plan(entries, hours=10.0, seed=0)
        self.assertEqual([e["video_id"] for e in got], ["v1"])

    def test_no_cover_entries_gives_an_empty_plan(self):
        self.assertEqual(self._plan(self._entries(5, "official"), hours=1.0, seed=0), [])

    def test_a_non_positive_budget_is_rejected(self):
        with self.assertRaises(ValueError):
            self._plan(self._entries(5), hours=0.0, seed=0)
