# backend/services/live_vocoder_analyzer.py

"""
Live-MIC-only acoustic forensic analyzer for LinguaShield.

IMPORTANT ARCHITECTURAL BOUNDARY
--------------------------------
This module is ONLY for the Live Intercept microphone pipeline.

It must NOT be imported by:
- Acoustic Batch Analysis
- Inject Audio File
- Biometric 1v1
- Smart Incident Export
- Lui
- any static/file-analysis workflow

The existing:
    backend/services/vocoder_analyzer.py
remains untouched.

This module is intentionally stateless at the analyzer level.
Per-WebSocket buffering, VAD state, EMA state, hysteresis state, and
session lifecycle belong in live_router.py, not here.

This analyzer receives a complete Live analysis window and returns
window-local forensic telemetry.

It does NOT claim calibrated AI/RVC detection accuracy.
"""

from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

import numpy as np
import scipy.signal as ssignal

logger = logging.getLogger("LinguaShield.LiveVocoderAnalyzer")


# ============================================================================
# CONFIGURATION
# ============================================================================

TARGET_SAMPLE_RATE = 16_000

# Minimum usable analysis duration.
MIN_ANALYSIS_SECONDS = 0.75
MIN_ANALYSIS_SAMPLES = int(TARGET_SAMPLE_RATE * MIN_ANALYSIS_SECONDS)

# Recommended Live analysis window.
# The router is responsible for constructing this window.
RECOMMENDED_WINDOW_SECONDS = 1.5
RECOMMENDED_WINDOW_SAMPLES = int(
    TARGET_SAMPLE_RATE * RECOMMENDED_WINDOW_SECONDS
)

# Speech-energy safety floor.
# This is a starting engineering threshold, not a universal acoustic constant.
DEFAULT_VAD_RMS_THRESHOLD = 0.010

# Small numerical constants.
EPSILON = 1e-8


# ============================================================================
# RESULT MODEL
# ============================================================================

@dataclass(frozen=True)
class LiveVocoderAnalysisResult:
    """
    Result produced for ONE Live-MIC analysis window.

    `synthetic_probability` is an engineering score in [0, 1].
    It is NOT a statistically calibrated probability unless the underlying
    detector has been calibrated externally.

    The Live router may apply temporal aggregation / EMA / hysteresis
    after receiving this result.
    """

    is_synthetic_voice: bool
    synthetic_probability: float
    confidence_label: str

    inference_backend: str = "live_acoustic_heuristic"

    latency_ms: float = 0.0
    clip_duration_secs: float = 0.0

    feature_diagnostics: Dict[str, Any] = field(default_factory=dict)

    stream_state: str = "inconclusive"

    warning: Optional[str] = None


# ============================================================================
# LIVE ANALYZER
# ============================================================================

class LiveVocoderAnalyzer:
    """
    Stateless analyzer for the Live microphone path.

    DESIGN PRINCIPLES
    -----------------
    1. No session state.
    2. No global rolling buffer.
    3. No EMA.
    4. No hysteresis.
    5. No transcript handling.
    6. No WebSocket handling.
    7. No interaction with the static/file analyzer.

    The caller supplies an already-constructed analysis window.
    """

    def __init__(
        self,
        *,
        target_sample_rate: int = TARGET_SAMPLE_RATE,
        vad_rms_threshold: float = DEFAULT_VAD_RMS_THRESHOLD,
    ) -> None:
        self.target_sample_rate = int(target_sample_rate)
        self.vad_rms_threshold = float(vad_rms_threshold)

        if self.target_sample_rate <= 0:
            raise ValueError("target_sample_rate must be positive")

        if self.vad_rms_threshold < 0:
            raise ValueError("vad_rms_threshold cannot be negative")

    # ------------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------------

    def analyze(
        self,
        source: Union[
            bytes,
            bytearray,
            memoryview,
            np.ndarray,
            io.BytesIO,
        ],
        *,
        sample_rate: int = TARGET_SAMPLE_RATE,
    ) -> LiveVocoderAnalysisResult:
        """
        Analyze one Live-MIC audio window.

        Accepted input:
            - PCM16 bytes
            - bytearray
            - memoryview
            - float/int numpy array
            - BytesIO containing PCM16 data

        Expected canonical format:
            mono
            16-bit PCM
            16 kHz

        The method itself is stateless.
        """

        started = time.perf_counter()

        try:
            samples = self._coerce_audio_to_float32(
                source,
                sample_rate=sample_rate,
            )

            samples = self._sanitize_samples(samples)

            if samples.size == 0:
                return self._result(
                    probability=0.0,
                    is_synthetic=False,
                    confidence_label="LOW",
                    stream_state="idle",
                    started=started,
                    samples=samples,
                    warning="Empty audio window.",
                )

            if sample_rate != self.target_sample_rate:
                samples = self._resample_if_needed(
                    samples,
                    source_rate=sample_rate,
                    target_rate=self.target_sample_rate,
                )

            rms = self._calculate_rms(samples)
            peak = float(np.max(np.abs(samples))) if samples.size else 0.0

            duration = (
                samples.size / float(self.target_sample_rate)
                if samples.size
                else 0.0
            )

            # --------------------------------------------------------------
            # LIVE VAD
            # --------------------------------------------------------------
            #
            # This module only evaluates the window itself.
            # The router remains responsible for persistent VAD state,
            # enter/exit hysteresis, and stream-level IDLE transitions.
            #
            if rms < self.vad_rms_threshold:
                return self._result(
                    probability=0.0,
                    is_synthetic=False,
                    confidence_label="LOW",
                    stream_state="idle",
                    started=started,
                    samples=samples,
                    diagnostics={
                        "rms": rms,
                        "peak": peak,
                        "vad_active": False,
                        "sample_count": int(samples.size),
                        "sample_rate": self.target_sample_rate,
                    },
                    warning="Insufficient speech energy.",
                )

            # --------------------------------------------------------------
            # MINIMUM WINDOW VALIDATION
            # --------------------------------------------------------------
            if samples.size < MIN_ANALYSIS_SAMPLES:
                return self._result(
                    probability=0.0,
                    is_synthetic=False,
                    confidence_label="LOW",
                    stream_state="inconclusive",
                    started=started,
                    samples=samples,
                    diagnostics={
                        "rms": rms,
                        "peak": peak,
                        "vad_active": True,
                        "sample_count": int(samples.size),
                        "sample_rate": self.target_sample_rate,
                    },
                    warning=(
                        f"Analysis window too short: "
                        f"{samples.size} samples "
                        f"({duration:.3f}s)."
                    ),
                )

            # --------------------------------------------------------------
            # FEATURE EXTRACTION
            # --------------------------------------------------------------
            diagnostics = self._extract_features(samples)

            diagnostics["rms"] = rms
            diagnostics["peak"] = peak
            diagnostics["vad_active"] = True
            diagnostics["sample_count"] = int(samples.size)
            diagnostics["sample_rate"] = self.target_sample_rate
            diagnostics["duration_secs"] = duration

            # --------------------------------------------------------------
            # WINDOW-LOCAL HEURISTIC SCORE
            # --------------------------------------------------------------
            probability, explanation = self._estimate_synthetic_score(
                diagnostics
            )

            diagnostics["decision_explanation"] = explanation

            is_synthetic = probability >= 0.65

            if probability >= 0.65:
                stream_state = "ai"
                confidence_label = "HIGH"
            elif probability <= 0.35:
                stream_state = "human"
                confidence_label = "HIGH"
            else:
                stream_state = "inconclusive"
                confidence_label = "MEDIUM"

            return self._result(
                probability=probability,
                is_synthetic=is_synthetic,
                confidence_label=confidence_label,
                stream_state=stream_state,
                started=started,
                samples=samples,
                diagnostics=diagnostics,
            )

        except Exception as exc:
            logger.exception("Live-MIC analysis failure")

            return self._result(
                probability=0.0,
                is_synthetic=False,
                confidence_label="LOW",
                stream_state="inconclusive",
                started=started,
                samples=np.empty(0, dtype=np.float32),
                warning=str(exc),
            )

    def analyze_audio_bytes(
        self,
        raw_pcm: bytes,
        *,
        sample_rate: int = TARGET_SAMPLE_RATE,
    ) -> LiveVocoderAnalysisResult:
        """
        Explicit PCM16 convenience method for live_router.py.
        """

        return self.analyze(
            raw_pcm,
            sample_rate=sample_rate,
        )

    # ------------------------------------------------------------------------
    # INPUT NORMALIZATION
    # ------------------------------------------------------------------------

    def _coerce_audio_to_float32(
        self,
        source: Union[
            bytes,
            bytearray,
            memoryview,
            np.ndarray,
            io.BytesIO,
        ],
        *,
        sample_rate: int,
    ) -> np.ndarray:
        del sample_rate  # Explicitly unused here; resampling happens later.

        if isinstance(source, io.BytesIO):
            source.seek(0)
            raw = source.read()
            return self._pcm16_bytes_to_float32(raw)

        if isinstance(source, (bytes, bytearray, memoryview)):
            return self._pcm16_bytes_to_float32(bytes(source))

        if isinstance(source, np.ndarray):
            array = np.asarray(source)

            if array.ndim == 0:
                return np.empty(0, dtype=np.float32)

            # Accept mono shape [N].
            if array.ndim == 1:
                values = array
            # Accept [N, 1].
            elif array.ndim == 2 and array.shape[1] == 1:
                values = array[:, 0]
            # Accept [1, N].
            elif array.ndim == 2 and array.shape[0] == 1:
                values = array[0]
            else:
                raise ValueError(
                    f"Live analyzer expects mono audio, got shape={array.shape}"
                )

            if np.issubdtype(values.dtype, np.integer):
                info = np.iinfo(values.dtype)
                scale = max(abs(info.min), info.max)

                if scale <= 0:
                    raise ValueError("Invalid integer audio dtype")

                return values.astype(np.float32) / float(scale)

            return values.astype(np.float32, copy=False)

        raise TypeError(
            "Unsupported Live audio input type: "
            f"{type(source).__name__}"
        )

    @staticmethod
    def _pcm16_bytes_to_float32(raw: bytes) -> np.ndarray:
        if not raw:
            return np.empty(0, dtype=np.float32)

        if len(raw) % 2 != 0:
            raise ValueError(
                "PCM16 byte buffer has an odd byte length."
            )

        pcm16 = np.frombuffer(
            raw,
            dtype="<i2",
        )

        return pcm16.astype(np.float32) / 32768.0

    @staticmethod
    def _sanitize_samples(samples: np.ndarray) -> np.ndarray:
        result = np.asarray(samples, dtype=np.float32)

        result = np.nan_to_num(
            result,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        # Keep the analyzer numerically bounded.
        result = np.clip(result, -1.0, 1.0)

        return result

    @staticmethod
    def _resample_if_needed(
        samples: np.ndarray,
        *,
        source_rate: int,
        target_rate: int,
    ) -> np.ndarray:
        if source_rate <= 0:
            raise ValueError("source_rate must be positive")

        if source_rate == target_rate:
            return samples

        if samples.size == 0:
            return samples

        gcd = np.gcd(source_rate, target_rate)

        up = target_rate // gcd
        down = source_rate // gcd

        resampled = ssignal.resample_poly(
            samples,
            up,
            down,
        )

        return np.asarray(
            resampled,
            dtype=np.float32,
        )

    # ------------------------------------------------------------------------
    # BASIC AUDIO METRICS
    # ------------------------------------------------------------------------

    @staticmethod
    def _calculate_rms(samples: np.ndarray) -> float:
        if samples.size == 0:
            return 0.0

        value = np.sqrt(
            np.mean(
                np.square(samples, dtype=np.float32)
            )
        )

        return float(value)

    # ------------------------------------------------------------------------
    # FEATURE EXTRACTION
    # ------------------------------------------------------------------------

    def _extract_features(
        self,
        samples: np.ndarray,
    ) -> Dict[str, float]:
        """
        Extract lightweight deterministic acoustic diagnostics.

        These are diagnostic features, not individually authoritative
        indicators of synthetic audio.
        """

        sr = self.target_sample_rate

        # --------------------------------------------------------------
        # 25 ms analysis frames / 50% hop
        # --------------------------------------------------------------
        frame_len = max(1, int(sr * 0.025))
        hop_len = max(1, frame_len // 2)

        if samples.size < frame_len:
            padded = np.pad(
                samples,
                (0, frame_len - samples.size),
            )
            frames = np.asarray([padded], dtype=np.float32)
        else:
            starts = np.arange(
                0,
                samples.size - frame_len + 1,
                hop_len,
            )

            frames = np.stack(
                [
                    samples[start:start + frame_len]
                    for start in starts
                ],
                axis=0,
            ).astype(np.float32, copy=False)

        window = np.hanning(frame_len).astype(np.float32)

        # --------------------------------------------------------------
        # ZCR
        # --------------------------------------------------------------
        signs = np.signbit(frames)

        zcr_values = (
            np.mean(
                signs[:, 1:] != signs[:, :-1],
                axis=1,
            )
            if frames.shape[1] > 1
            else np.zeros(frames.shape[0], dtype=np.float32)
        )

        zcr_mean = float(np.mean(zcr_values))
        zcr_std = float(np.std(zcr_values))

        zcr_dispersion = float(
            zcr_std / (zcr_mean + EPSILON)
        )

        # --------------------------------------------------------------
        # Frame energy
        # --------------------------------------------------------------
        frame_energy = np.mean(
            np.square(frames),
            axis=1,
        )

        mean_energy = float(
            np.mean(frame_energy)
        )

        energy_std = float(
            np.std(frame_energy)
        )

        energy_cv = float(
            energy_std /
            (mean_energy + EPSILON)
        )

        # --------------------------------------------------------------
        # Spectral features
        # --------------------------------------------------------------
        fft_frames = np.abs(
            np.fft.rfft(
                frames * window,
                axis=1,
            )
        ).astype(np.float32)

        spectral_flux = 0.0
        spectral_flatness = 0.0
        spectral_centroid = 0.0
        high_frequency_ratio = 0.0

        if fft_frames.shape[0] >= 2:
            diff = np.diff(
                fft_frames,
                axis=0,
            )

            denominator = float(
                np.mean(fft_frames[:-1])
            ) + EPSILON

            spectral_flux = float(
                np.mean(np.abs(diff)) /
                denominator
            )

        # --------------------------------------------------------------
        # Spectral flatness
        # --------------------------------------------------------------
        magnitude = fft_frames + EPSILON

        geometric_mean = np.exp(
            np.mean(
                np.log(magnitude),
                axis=1,
            )
        )

        arithmetic_mean = np.mean(
            magnitude,
            axis=1,
        )

        flatness_values = (
            geometric_mean /
            (arithmetic_mean + EPSILON)
        )

        spectral_flatness = float(
            np.mean(flatness_values)
        )

        # --------------------------------------------------------------
        # Spectral centroid
        # --------------------------------------------------------------
        frequencies = np.fft.rfftfreq(
            frame_len,
            d=1.0 / sr,
        ).astype(np.float32)

        weighted_sum = np.sum(
            fft_frames * frequencies[None, :],
            axis=1,
        )

        total_power = np.sum(
            fft_frames,
            axis=1,
        ) + EPSILON

        centroid_values = (
            weighted_sum / total_power
        )

        spectral_centroid = float(
            np.mean(centroid_values)
        )

        # --------------------------------------------------------------
        # High-frequency energy ratio
        # --------------------------------------------------------------
        split_hz = 4000.0

        split_index = int(
            np.searchsorted(
                frequencies,
                split_hz,
            )
        )

        split_index = max(
            1,
            min(
                split_index,
                fft_frames.shape[1] - 1,
            ),
        )

        low_energy = np.sum(
            fft_frames[:, :split_index],
            axis=1,
        )

        high_energy = np.sum(
            fft_frames[:, split_index:],
            axis=1,
        )

        high_frequency_ratio = float(
            np.mean(
                high_energy /
                (
                    low_energy +
                    high_energy +
                    EPSILON
                )
            )
        )

        return {
            "zcr_mean": zcr_mean,
            "zcr_std": zcr_std,
            "zcr_dispersion": zcr_dispersion,
            "energy_mean": mean_energy,
            "energy_std": energy_std,
            "energy_coefficient_of_variation": energy_cv,
            "spectral_flux": spectral_flux,
            "spectral_flatness": spectral_flatness,
            "spectral_centroid_hz": spectral_centroid,
            "high_frequency_ratio": high_frequency_ratio,
        }

    # ------------------------------------------------------------------------
    # LIVE WINDOW SCORING
    # ------------------------------------------------------------------------

    @staticmethod
    def _estimate_synthetic_score(
        features: Dict[str, float],
    ) -> tuple[float, list[str]]:
        """
        Convert acoustic diagnostics into a conservative engineering score.

        IMPORTANT:
        This is NOT a calibrated ML probability.

        It intentionally requires multiple acoustic indications instead of
        using one feature as an absolute "AI" detector.

        This prevents the Live frontend from pretending that one feature,
        such as spectral flux alone, proves synthetic speech.
        """

        energy_cv = float(
            features.get(
                "energy_coefficient_of_variation",
                0.0,
            )
        )

        spectral_flux = float(
            features.get(
                "spectral_flux",
                0.0,
            )
        )

        zcr_dispersion = float(
            features.get(
                "zcr_dispersion",
                0.0,
            )
        )

        spectral_flatness = float(
            features.get(
                "spectral_flatness",
                0.0,
            )
        )

        high_frequency_ratio = float(
            features.get(
                "high_frequency_ratio",
                0.0,
            )
        )

        score = 0.10
        reasons: list[str] = []

        # --------------------------------------------------------------
        # Acoustic regularity
        # --------------------------------------------------------------
        if energy_cv < 0.25:
            score += 0.20
            reasons.append(
                "unusually low frame-energy variation"
            )
        elif energy_cv < 0.35:
            score += 0.10
            reasons.append(
                "reduced frame-energy variation"
            )

        # --------------------------------------------------------------
        # Spectral dynamics
        # --------------------------------------------------------------
        if spectral_flux < 0.10:
            score += 0.25
            reasons.append(
                "very low spectral flux"
            )
        elif spectral_flux < 0.18:
            score += 0.12
            reasons.append(
                "reduced spectral flux"
            )

        # --------------------------------------------------------------
        # ZCR consistency
        # --------------------------------------------------------------
        if zcr_dispersion > 0.45:
            score += 0.12
            reasons.append(
                "high zero-crossing dispersion"
            )

        # --------------------------------------------------------------
        # Spectral flatness
        # --------------------------------------------------------------
        if spectral_flatness < 0.10:
            score += 0.10
            reasons.append(
                "low spectral flatness"
            )

        # --------------------------------------------------------------
        # Extremely low high-frequency content
        # --------------------------------------------------------------
        #
        # This is deliberately weak because codecs, microphones, and
        # telephony can also remove high frequencies.
        #
        if high_frequency_ratio < 0.04:
            score += 0.05
            reasons.append(
                "low high-frequency energy ratio"
            )

        # --------------------------------------------------------------
        # Combined pattern
        # --------------------------------------------------------------
        if (
            energy_cv < 0.35
            and spectral_flux < 0.18
        ):
            score += 0.10
            reasons.append(
                "combined temporal/spectral regularity"
            )

        score = float(
            np.clip(
                score,
                0.0,
                0.98,
            )
        )

        return score, reasons

    # ------------------------------------------------------------------------
    # RESULT BUILDER
    # ------------------------------------------------------------------------

    def _result(
        self,
        *,
        probability: float,
        is_synthetic: bool,
        confidence_label: str,
        stream_state: str,
        started: float,
        samples: np.ndarray,
        diagnostics: Optional[Dict[str, Any]] = None,
        warning: Optional[str] = None,
    ) -> LiveVocoderAnalysisResult:
        latency_ms = (
            time.perf_counter() - started
        ) * 1000.0

        duration = (
            samples.size /
            float(self.target_sample_rate)
            if samples.size
            else 0.0
        )

        return LiveVocoderAnalysisResult(
            is_synthetic_voice=bool(is_synthetic),
            synthetic_probability=float(
                np.clip(
                    probability,
                    0.0,
                    1.0,
                )
            ),
            confidence_label=confidence_label,
            inference_backend="live_acoustic_heuristic",
            latency_ms=round(
                latency_ms,
                2,
            ),
            clip_duration_secs=round(
                duration,
                3,
            ),
            feature_diagnostics=diagnostics or {},
            stream_state=stream_state,
            warning=warning,
        )


__all__ = [
    "LiveVocoderAnalyzer",
    "LiveVocoderAnalysisResult",
    "TARGET_SAMPLE_RATE",
    "RECOMMENDED_WINDOW_SECONDS",
    "RECOMMENDED_WINDOW_SAMPLES",
]