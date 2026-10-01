"""実物の content encoder と F0 抽出器を、[`extract.py`](extract.py) が受け取る形に合わせる薄い層。

ここだけが重いモデルに触れます。[`extract.py`](extract.py) と [`shard.py`](shard.py) は
呼び出し可能オブジェクトを受け取るだけなので、単体テストはモデルもネットワークも要りません。

**content encoder の決定:** ContentVec（`lengyue233/content-vec-best`、MIT、768 次元、layer 12）。
根拠は [content encoder の選定](../../doc/svc-content-encoder.md)。話者情報の除去を目的に
追加学習されており、SVC で最大のリスクである timbre 漏れに直接効きます。

**F0 の決定:** RMVPE 固定。リポジトリの `preprocess/f0_rmvpe.py` は mel と同じフレーム数を
返すので（実測で確認済み）、そのまま使えます。
"""
from __future__ import annotations

import functools
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_CONTENTVEC = "lengyue233/content-vec-best"
CONTENTVEC_SR = 16000
CONTENTVEC_STRIDE = 320          # 16 kHz / 320 = 50 Hz

# RMVPE には release version が無く、初回実行時に落ちてくる重み 1 個が実体です。
# 「F0 extractor の version」（M1 ゴール 5）はこの重みの checksum と入手元で表します。
RMVPE_WEIGHT_PATH = Path(__file__).resolve().parents[1] / "algorithms" / "rmvpe.pt"
RMVPE_WEIGHT_URL = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt"


class ContentVecEncoder:
    """ContentVec の指定層の隠れ状態を `[T_ssl, C]` で返す。

    `extract_phrase(content_encoder=...)` にそのまま渡せます。**16 kHz 前提**で、
    それ以外の sample rate を渡されたら例外にします（黙って resample すると、
    どの rate で抽出したのかが manifest と食い違うため）。
    """

    def __init__(self, model_id: str = DEFAULT_CONTENTVEC, *, layer: int = 12,
                 device: str = "cpu", revision: str | None = None):
        import torch
        from transformers import HubertModel

        self.model_id = model_id
        self.layer = int(layer)
        self.device = device
        self.revision = revision
        self._torch = torch
        self.model = HubertModel.from_pretrained(model_id, revision=revision).to(device).eval()
        n_layers = int(getattr(self.model.config, "num_hidden_layers", 0))
        if not 0 <= self.layer <= n_layers:
            raise ValueError(f"layer {self.layer} は 0..{n_layers} の範囲外です")

    def __call__(self, wav: np.ndarray, sr: int) -> np.ndarray:
        if int(sr) != CONTENTVEC_SR:
            raise ValueError(f"ContentVec は {CONTENTVEC_SR} Hz 前提です; got {sr}")
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim != 1:
            raise ValueError(f"wav must be mono 1-D; got shape {wav.shape}")
        with self._torch.no_grad():
            x = self._torch.from_numpy(wav)[None].to(self.device)
            out = self.model(x, output_hidden_states=True)
        # hidden_states[0] は embedding 出力、hidden_states[i] が i 層目の出力。
        return out.hidden_states[self.layer][0].float().cpu().numpy().astype(np.float32)

    def manifest(self) -> dict[str, Any]:
        """M1 ゴール 5 が要求する再現情報。抽出条件は後段の全実験の比較基盤になる。"""
        return {
            "content_encoder": self.model_id,
            "content_encoder_revision": self.revision or "main",
            "content_encoder_layer": self.layer,
            "content_encoder_sr": CONTENTVEC_SR,
            "content_encoder_stride": CONTENTVEC_STRIDE,
            "content_encoder_hidden": int(self.model.config.hidden_size),
        }


class NativeGridClip:
    """RMVPE の**ネイティブ格子**で、有声フレームの F0 を `[fmin, fmax]` に clip する mixin。

    上流は 9ceca86 でこの clip を外しました（範囲外の値が捨てられずに境界へ張り付き、
    平坦な音を作るため）。SVS はそれに従います。**SVC は既存の shard と checkpoint が
    この clip 込みの F0 で作られているので、bit 一致を保つために残します。**

    旧実装の clip は `_sanity_check`、つまり mel 格子への補間の**前**に掛かっていました。
    `extract_f0_rmvpe` の結果を後から clip しても一致しないので、同じ段に差し込みます。
    """

    def _sanity_check(self, pitch, periodicity):
        pitch, periodicity = super()._sanity_check(pitch, periodicity)
        voiced = periodicity > 0
        pitch[voiced] = np.clip(pitch[voiced], self.fmin, self.fmax)
        return pitch, periodicity


@functools.cache
def clipped_rmvpe_class():
    """`NativeGridClip` を掛けた RMVPE。torch を import するので初回呼び出しまで遅らせる。"""
    from preprocess.algorithms.rmvpe import RMVPEPitchAlgorithm

    class ClippedRMVPE(NativeGridClip, RMVPEPitchAlgorithm):
        pass

    return ClippedRMVPE


# 既定の RMVPE は重み（181 MB）を読むので、同じ条件なら使い回す（旧 `_get_algo` と同じ扱い）。
_CLIPPED_CACHE: dict = {}


def _default_algo(**kw):
    key = tuple(sorted(kw.items()))
    if key not in _CLIPPED_CACHE:
        _CLIPPED_CACHE[key] = clipped_rmvpe_class()(**kw)
    return _CLIPPED_CACHE[key]


class RmvpeF0:
    """`extract_phrase(f0_extract=...)` の形に合わせた RMVPE の薄い包み。

    **確認済み:** `extract_f0_rmvpe` は mel と同じフレーム数を返します（44,100 / 30,011 /
    65,537 サンプルで実測、差 0）。`interpolate=False` で無声を 0 のまま返し、
    `uv` を別に受け取ります。F0 は `NativeGridClip` で `[f0_min, f0_max]` に clip します。

    `algo_factory` は単体テスト用の口です（RMVPE の重みを読まずに clip の段を試すため）。
    """

    def __init__(self, *, f0_min: float = 65.0, f0_max: float = 1100.0,
                 device: str = "cpu", algo_factory=None):
        self.f0_min, self.f0_max, self.device = float(f0_min), float(f0_max), device
        self._algo_factory = algo_factory or _default_algo

    def __call__(self, wav: np.ndarray, sr: int, hop: int):
        from preprocess.f0_rmvpe import extract_f0_rmvpe
        algo = self._algo_factory(sample_rate=int(sr), hop_size=int(hop), fmin=self.f0_min,
                                  fmax=self.f0_max, device=self.device)
        f0, uv = extract_f0_rmvpe(wav, int(sr), int(hop), device=self.device,
                                  interpolate=False, algo=algo)
        return np.asarray(f0, np.float32), np.asarray(uv, np.float32)

    def manifest(self) -> dict[str, Any]:
        """M1 ゴール 5 の「F0 extractor の version」。

        `"rmvpe"` という名前は version になりません。重みは初回実行時に落ちてくるので、
        **重みそのものの sha256 と入手元**を残します。未取得の環境では checksum を
        `None` にして、manifest 生成自体は落としません（1 段目を回せば必ず存在します）。
        """
        digest = None
        if RMVPE_WEIGHT_PATH.exists():
            h = hashlib.sha256()
            with RMVPE_WEIGHT_PATH.open("rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            digest = h.hexdigest()
        return {"f0_extractor": "rmvpe", "f0_min": self.f0_min, "f0_max": self.f0_max,
                "f0_extractor_weight_url": RMVPE_WEIGHT_URL,
                "f0_extractor_weight_sha256": digest}
