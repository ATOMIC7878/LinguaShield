# backend/routers/live_router.py

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional
from pathlib import Path

import numpy as np
import torch
from speechbrain.inference.speaker import EncoderClassifier
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from services.live_vocoder_analyzer import (
    RECOMMENDED_WINDOW_SAMPLES,
    TARGET_SAMPLE_RATE,
)
from services.vocoder_analyzer import DeepfakeVocoderAnalyzer


# ---------------------------------------------------------------------------
# LIVE-MIC SPEAKER ID TRACKING (SESSION-LOCAL)
# ---------------------------------------------------------------------------
# This is intentionally isolated to Live MIC. It does not modify the frozen
# 1v1 Biometric verifier or any static Acoustic workflow.
LIVE_SPEAKER_MATCH_THRESHOLD = 0.60
LIVE_SPEAKER_UPDATE_SECONDS = 1.5
LIVE_SPEAKER_CONFIRMATIONS = 4
LIVE_NEW_SPEAKER_MAX_SIMILARITY = 0.50

_live_speaker_encoder = None


def _get_live_speaker_encoder() -> EncoderClassifier:
    """Load the same local ECAPA speaker encoder used by 1v1, once per process."""
    global _live_speaker_encoder
    if _live_speaker_encoder is not None:
        return _live_speaker_encoder

    # Use the same local model location as the frozen 1v1 Biometric verifier.
    # live_router.py lives in backend/routers/, so parents[1] is backend/.
    # live_router.py is directly inside backend/, so parents[0] is backend/.
    # This matches the actual LinguaShield project layout:
    # backend/models/speaker_encoder/spkrec-ecapa-voxceleb
    model_dir = (
        Path(__file__).resolve().parents[0]
        / "models"
        / "speaker_encoder"
        / "spkrec-ecapa-voxceleb"
    )

    if not model_dir.exists():
        # Compatibility fallback for deployments where the model is stored
        # at the project root instead of under backend/.
        fallback_dir = (
            Path(__file__).resolve().parents[1]
            / "models"
            / "speaker_encoder"
            / "spkrec-ecapa-voxceleb"
        )
        if fallback_dir.exists():
            model_dir = fallback_dir
        else:
            raise FileNotFoundError(
                f"ECAPA-TDNN model directory not found: {model_dir}"
            )

    logger.info(
        "[LiveSpeaker] Loading local ECAPA-TDNN speaker encoder from: %s",
        model_dir,
    )
    _live_speaker_encoder = EncoderClassifier.from_hparams(
        source=str(model_dir),
        savedir=str(model_dir),
        run_opts={"device": "cpu"},
    )
    logger.info("[LiveSpeaker] ECAPA-TDNN speaker encoder loaded successfully.")
    return _live_speaker_encoder


def _normalize_live_embedding(embedding: np.ndarray) -> np.ndarray:
    vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(vec)
    if vec.size == 0 or norm <= 1e-10:
        raise ValueError("Invalid Live speaker embedding")
    return (vec / norm).astype(np.float32)


def _extract_live_speaker_embedding(pcm16: np.ndarray) -> np.ndarray:
    """Create a normalized ECAPA embedding from a 16 kHz mono PCM16 window."""
    samples = np.asarray(pcm16, dtype=np.float32).reshape(-1) / 32768.0
    if samples.size < int(TARGET_SR * 0.75):
        raise ValueError("Insufficient audio for Live speaker embedding")

    samples = samples - float(np.mean(samples))
    peak = float(np.max(np.abs(samples)))
    if peak <= 1e-10:
        raise ValueError("Silent audio cannot produce a speaker embedding")
    samples = samples / peak

    waveform = torch.from_numpy(samples.astype(np.float32)).unsqueeze(0)
    encoder = _get_live_speaker_encoder()

    with torch.inference_mode():
        embedding = encoder.encode_batch(waveform)

    return _normalize_live_embedding(
        embedding.detach().cpu().numpy()
    )


def _live_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = _normalize_live_embedding(a)
    b = _normalize_live_embedding(b)
    return float(np.clip(np.dot(a, b), 0.0, 1.0))


class LiveSpeakerRegistry:
    """Session-local persistent speaker identity map."""

    def __init__(
        self,
        match_threshold: float = LIVE_SPEAKER_MATCH_THRESHOLD,
        required_confirmations: int = LIVE_SPEAKER_CONFIRMATIONS,
    ) -> None:
        self.match_threshold = match_threshold
        self.required_confirmations = required_confirmations
        self.profiles: dict[int, np.ndarray] = {}
        self.next_speaker_id = 1
        self.active_speaker_id: Optional[int] = None
        self.candidate_speaker_id: Optional[int] = None
        self.candidate_hits = 0

    def assign(self, embedding: np.ndarray) -> tuple[int, float, bool]:
        """Return (speaker_id, best_similarity, changed).

        The active speaker is intentionally sticky. A new identity is only
        committed after repeated evidence across consecutive analysis windows.
        This prevents normal ECAPA variation within one person's speech from
        creating VOICE 2 / VOICE 3.
        """
        if not self.profiles:
            speaker_id = self.next_speaker_id
            self.next_speaker_id += 1
            self.profiles[speaker_id] = embedding
            self.active_speaker_id = speaker_id
            self.candidate_speaker_id = None
            self.candidate_hits = 0
            return speaker_id, 1.0, True

        best_id = None
        best_similarity = -1.0
        for speaker_id, profile in self.profiles.items():
            similarity = _live_cosine_similarity(embedding, profile)
            if similarity > best_similarity:
                best_similarity = similarity
                best_id = speaker_id

        active_id = self.active_speaker_id

        # ---------------------------------------------------------------
        # 1) Current window still matches a known speaker.
        # ---------------------------------------------------------------
        if best_id is not None and best_similarity >= self.match_threshold:
            if best_id == active_id:
                # Same active speaker: keep identity immediately and clear
                # any pending switch candidate caused by transient variation.
                self.candidate_speaker_id = None
                self.candidate_hits = 0

                # Slow profile adaptation prevents one window from moving the
                # reference too aggressively.
                return best_id, best_similarity, False

            # A different known profile must remain stable for several
            # consecutive windows before we switch the active speaker.
            if self.candidate_speaker_id == best_id:
                self.candidate_hits += 1
            else:
                self.candidate_speaker_id = best_id
                self.candidate_hits = 1

            if self.candidate_hits >= self.required_confirmations:
                changed = active_id != best_id
                self.active_speaker_id = best_id
                self.candidate_speaker_id = None
                self.candidate_hits = 0

                return best_id, best_similarity, changed

            # Do not split the transcript while the candidate is still
            # unconfirmed; preserve the current speaker identity.
            return active_id or best_id, best_similarity, False

        # ---------------------------------------------------------------
        # 2) No known profile is close enough.
        # ---------------------------------------------------------------
        # Failing the 0.60 match threshold alone is not enough to create a
        # new speaker. Require a clearly low similarity and repeated evidence.
        if best_similarity > LIVE_NEW_SPEAKER_MAX_SIMILARITY:
            self.candidate_speaker_id = None
            self.candidate_hits = 0
            return active_id or best_id or self.next_speaker_id, best_similarity, False

        # Live MIC demo is intentionally limited to two persistent speakers.
        # Once VOICE 1 and VOICE 2 exist, do not create VOICE 3 from a noisy
        # or low-confidence ECAPA window; keep the currently active identity.
        if len(self.profiles) >= 2:
            self.candidate_speaker_id = None
            self.candidate_hits = 0
            return active_id or best_id or self.next_speaker_id, best_similarity, False

        new_id = self.next_speaker_id

        if self.candidate_speaker_id == new_id:
            self.candidate_hits += 1
        else:
            self.candidate_speaker_id = new_id
            self.candidate_hits = 1

        if self.candidate_hits >= self.required_confirmations:
            self.next_speaker_id += 1
            self.profiles[new_id] = embedding
            self.active_speaker_id = new_id
            self.candidate_speaker_id = None
            self.candidate_hits = 0
            return new_id, best_similarity, True

        return active_id or new_id, best_similarity, False


logger = logging.getLogger("LinguaShield.LiveRouter")

router = APIRouter(prefix="/api/live", tags=["Live Intercept"])

# ---------------------------------------------------------------------------
# LIVE-MIC ONLY CONFIGURATION
# ---------------------------------------------------------------------------

TARGET_SR = TARGET_SAMPLE_RATE
LIVE_WINDOW_SAMPLES = RECOMMENDED_WINDOW_SAMPLES  # 1.5 s @ 16 kHz
MIN_ANALYSIS_SAMPLES = int(TARGET_SR * 0.75)

# Browser currently sends ~4096 PCM16 samples per ScriptProcessor callback.
EXPECTED_CHUNK_SAMPLES = 4096

# Mic-only RMS VAD. These are engineering starting values, not universal
# acoustic constants.
VAD_ENTER_RMS = 0.010
VAD_EXIT_RMS = 0.008
SILENCE_TO_IDLE_SECONDS = 1.5

# Per-connection temporal stabilization.
EMA_ALPHA = 0.35
AI_ENTER_THRESHOLD = 0.70
HUMAN_ENTER_THRESHOLD = 0.35
REQUIRED_CONFIRMATIONS = 3


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------


def _clamp01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    x = np.asarray(samples, dtype=np.float32)
    return float(np.sqrt(np.mean(np.square(x))))


async def _send_idle(websocket: WebSocket) -> None:
    await websocket.send_json(
        {
            "status": "listening",
            "stream_state": "idle",
            "risk_classification": "AWAITING ACOUSTIC SIGNAL...",
            "insufficient_data": True,
            "is_scam": False,
            "confidence": 0.0,
            "synthetic_probability": 0.0,
            "is_synthetic_voice": False,
            "acoustic_forensics": {
                "vocoder_score": 0.0,
                "is_synthetic_voice": False,
                "vad_active": False,
            },
            "manipulation_vectors": {
                "urgency": 0,
                "authority": 0,
                "isolation": 0,
            },
        }
    )


async def _send_warmup(
    websocket: WebSocket,
    *,
    rms_value: float,
    buffered_samples: int,
    ema_score: Optional[float],
) -> None:
    score = _clamp01(ema_score or 0.0)

    await websocket.send_json(
        {
            "status": "analyzing",
            "stream_state": "inconclusive",
            "risk_classification": "COLLECTING ACOUSTIC EVIDENCE...",
            "insufficient_data": True,
            "is_scam": False,
            "confidence": round(max(score, 1.0 - score) * 100.0, 2),
            "synthetic_probability": round(score, 4),
            "is_synthetic_voice": False,
            "acoustic_forensics": {
                "vocoder_score": round(score * 100.0, 2),
                "raw_vocoder_score": 0.0,
                "ema_vocoder_score": round(score * 100.0, 2),
                "is_synthetic_voice": False,
                "vad_active": True,
                "vad_rms": round(rms_value, 6),
                "buffered_samples": buffered_samples,
                "window_samples": LIVE_WINDOW_SAMPLES,
                "sample_rate": TARGET_SR,
            },
            "manipulation_vectors": {
                "urgency": 0,
                "authority": 0,
                "isolation": 0,
            },
        }
    )


# ---------------------------------------------------------------------------
# LIVE MIC WEBSOCKET
# ---------------------------------------------------------------------------


@router.websocket("/stream")
async def live_audio_stream(websocket: WebSocket) -> None:
    """
    Live microphone-only forensic pipeline.

    IMPORTANT:
    - Uses the validated Acoustic analyzer for synthetic scoring.
    - Uses a Live-only ECAPA speaker registry for persistent speaker IDs.
    - All stream buffers and stabilization state are per WebSocket.
    - Inject Audio File and other static workflows are untouched.
    """

    await websocket.accept()

    logger.info("LIVE INTERCEPT: WebSocket connected from %s", websocket.client)
    print("\n[+] LIVE INTERCEPT: Live-MIC forensic engine initialized and ready.")

    analyzer = DeepfakeVocoderAnalyzer()

    # Per-WebSocket audio state.
    audio_buffer = np.empty(0, dtype=np.int16)

    # Per-WebSocket temporal classifier state.
    ema_score: Optional[float] = None
    confirmed_state = "idle"
    candidate_state = "idle"
    candidate_hits = 0

    # Per-WebSocket VAD state.
    speech_active = False
    last_speech_time: Optional[float] = None

    # Latest diagnostics.
    latest_raw_score = 0.0
    latest_ema_score = 0.0

    # Per-connection persistent speaker identity state.
    speaker_registry = LiveSpeakerRegistry()
    last_speaker_embedding_time = 0.0
    latest_speaker_id: Optional[int] = None
    latest_speaker_similarity = 0.0

    # Helps guarantee stale inference results do not get committed after
    # a disconnect/connection lifecycle change.
    stream_generation = 0
    active = True

    try:
        while active:
            raw_bytes = await websocket.receive_bytes()

            if not raw_bytes:
                continue

            # ---------------------------------------------------------------
            # PCM16 DECODE
            # ---------------------------------------------------------------
            if len(raw_bytes) % 2 != 0:
                logger.warning(
                    "LIVE INTERCEPT: ignoring odd-length PCM16 packet (%d bytes)",
                    len(raw_bytes),
                )
                continue

            try:
                pcm_chunk = np.frombuffer(raw_bytes, dtype="<i2")
            except Exception:
                logger.exception("LIVE INTERCEPT: PCM16 decode failed")
                continue

            if pcm_chunk.size == 0:
                continue

            float_chunk = pcm_chunk.astype(np.float32, copy=False) / 32768.0
            chunk_rms = _rms(float_chunk)
            now = time.monotonic()

            # ---------------------------------------------------------------
            # MIC-ONLY VAD WITH HYSTERESIS
            # ---------------------------------------------------------------
            if not speech_active:
                if chunk_rms >= VAD_ENTER_RMS:
                    speech_active = True
                    last_speech_time = now
            else:
                if chunk_rms >= VAD_EXIT_RMS:
                    last_speech_time = now
                elif (
                    last_speech_time is not None
                    and now - last_speech_time >= SILENCE_TO_IDLE_SECONDS
                ):
                    speech_active = False
                    last_speech_time = None

                    # A new speech episode gets a clean forensic window.
                    audio_buffer = np.empty(0, dtype=np.int16)
                    ema_score = None
                    confirmed_state = "idle"
                    candidate_state = "idle"
                    candidate_hits = 0
                    latest_raw_score = 0.0
                    latest_ema_score = 0.0

                    await _send_idle(websocket)
                    continue

            # Silence / not-yet-speech never becomes HUMAN automatically.
            if not speech_active:
                await _send_idle(websocket)
                continue

            # ---------------------------------------------------------------
            # PER-CONNECTION FIFO / RING BUFFER
            # ---------------------------------------------------------------
            audio_buffer = np.concatenate((audio_buffer, pcm_chunk))

            if audio_buffer.size > LIVE_WINDOW_SAMPLES:
                audio_buffer = audio_buffer[-LIVE_WINDOW_SAMPLES:]

            buffered_samples = int(audio_buffer.size)

            # Warm-up phase: never make a forensic decision from a partial
            # window. Keep the frontend neutral while evidence accumulates.
            if buffered_samples < MIN_ANALYSIS_SAMPLES:
                await _send_warmup(
                    websocket,
                    rms_value=chunk_rms,
                    buffered_samples=buffered_samples,
                    ema_score=ema_score,
                )
                continue

            if buffered_samples < LIVE_WINDOW_SAMPLES:
                await _send_warmup(
                    websocket,
                    rms_value=chunk_rms,
                    buffered_samples=buffered_samples,
                    ema_score=ema_score,
                )
                continue

            # ---------------------------------------------------------------
            # SNAPSHOT WINDOW
            # ---------------------------------------------------------------
            analysis_window = np.asarray(
                audio_buffer[-LIVE_WINDOW_SAMPLES:],
                dtype=np.int16,
            ).copy()

            # ---------------------------------------------------------------
            # LIVE-SPECIFIC ANALYSIS
            # ---------------------------------------------------------------
            stream_generation += 1
            operation_generation = stream_generation

            try:
                result = await asyncio.to_thread(
                    analyzer.analyze,
                    analysis_window.tobytes(),
                )
            except Exception as exc:
                logger.exception("LIVE INTERCEPT: analyzer failure")

                # Fail closed for this window; never force HUMAN.
                confirmed_state = "inconclusive"
                candidate_state = "inconclusive"
                candidate_hits = 0

                await websocket.send_json(
                    {
                        "status": "analyzing",
                        "stream_state": "inconclusive",
                        "risk_classification": "ACOUSTIC ANALYSIS UNAVAILABLE",
                        "insufficient_data": True,
                        "is_scam": False,
                        "confidence": 0.0,
                        "synthetic_probability": 0.0,
                        "is_synthetic_voice": False,
                        "acoustic_forensics": {
                            "vocoder_score": 0.0,
                            "is_synthetic_voice": False,
                            "vad_active": True,
                            "vad_rms": round(chunk_rms, 6),
                            "analysis_window_samples": LIVE_WINDOW_SAMPLES,
                            "sample_rate": TARGET_SR,
                        },
                        "manipulation_vectors": {
                            "urgency": 0,
                            "authority": 0,
                            "isolation": 0,
                        },
                        "warning": str(exc),
                    }
                )
                continue

            # If the stream was invalidated while inference was running,
            # discard the result instead of allowing a stale classification.
            if operation_generation != stream_generation or not active:
                continue

            # Analyzer-level silence/inconclusive result.
            if getattr(result, "stream_state", None) == "idle":
                await _send_idle(websocket)
                continue

            # ---------------------------------------------------------------
            # SESSION-LOCAL SPEAKER IDENTITY TRACKING
            # ---------------------------------------------------------------
            # Speaker identity persists across silence and forensic state
            # changes for the lifetime of this WebSocket session.
            speaker_changed = False
            if (
                time.monotonic() - last_speaker_embedding_time
                >= LIVE_SPEAKER_UPDATE_SECONDS
            ):
                try:
                    speaker_embedding = await asyncio.to_thread(
                        _extract_live_speaker_embedding,
                        analysis_window,
                    )
                    (
                        latest_speaker_id,
                        latest_speaker_similarity,
                        speaker_changed,
                    ) = speaker_registry.assign(speaker_embedding)
                    last_speaker_embedding_time = time.monotonic()
                except Exception as speaker_exc:
                    logger.warning(
                        "LIVE INTERCEPT: speaker tracking unavailable for this window: %s",
                        speaker_exc,
                    )

            raw_score = _clamp01(result.synthetic_probability)
            latest_raw_score = raw_score

            # ---------------------------------------------------------------
            # EMA
            # ---------------------------------------------------------------
            if ema_score is None:
                ema_score = raw_score
            else:
                ema_score = (
                    EMA_ALPHA * raw_score
                    + (1.0 - EMA_ALPHA) * ema_score
                )

            ema_score = _clamp01(ema_score)
            latest_ema_score = ema_score

            # ---------------------------------------------------------------
            # 4-STATE HYSTERESIS
            # ---------------------------------------------------------------
            if ema_score >= AI_ENTER_THRESHOLD:
                new_candidate = "ai"
            elif ema_score <= HUMAN_ENTER_THRESHOLD:
                new_candidate = "human"
            else:
                new_candidate = "inconclusive"

            if new_candidate == candidate_state:
                candidate_hits += 1
            else:
                candidate_state = new_candidate
                candidate_hits = 1

            if candidate_hits >= REQUIRED_CONFIRMATIONS:
                confirmed_state = candidate_state

            # ---------------------------------------------------------------
            # DISPLAY STATE
            # ---------------------------------------------------------------
            display_state = confirmed_state

            if display_state == "ai":
                risk_label = "SYNTHETIC VOICE INDICATORS DETECTED"
                confidence = round(ema_score * 100.0, 2)
                display_is_synthetic = True

            elif display_state == "human":
                risk_label = "LIVE HUMAN-LIKE SPEECH"
                confidence = round((1.0 - ema_score) * 100.0, 2)
                display_is_synthetic = False

            else:
                risk_label = "INCONCLUSIVE ACOUSTIC EVIDENCE"
                confidence = round(
                    max(ema_score, 1.0 - ema_score) * 100.0,
                    2,
                )
                display_is_synthetic = False

            # IMPORTANT:
            # Synthetic voice authenticity and scam/content threat are not
            # the same dimension. Existing scam analysis remains elsewhere.
            is_scam = False

            diagnostics = dict(result.feature_diagnostics or {})

            await websocket.send_json(
                {
                    "status": "analyzing",
                    "stream_state": display_state,
                    "risk_classification": risk_label,
                    "insufficient_data": False,
                    "is_scam": is_scam,
                    "confidence": confidence,
                    "synthetic_probability": round(ema_score, 4),
                    "is_synthetic_voice": display_is_synthetic,
                    "speaker_id": latest_speaker_id,
                    "speaker_label": (
                        f"VOICE {latest_speaker_id}"
                        if latest_speaker_id is not None
                        else None
                    ),
                    "speaker_similarity": round(
                        latest_speaker_similarity * 100.0, 2
                    ) if latest_speaker_id is not None else 0.0,
                    "speaker_changed": speaker_changed,
                    "acoustic_forensics": {
                        # Existing frontend gauge can consume this stabilized
                        # value instead of the raw per-window score.
                        "vocoder_score": round(ema_score * 100.0, 2),
                        "raw_vocoder_score": round(
                            latest_raw_score * 100.0,
                            2,
                        ),
                        "ema_vocoder_score": round(
                            latest_ema_score * 100.0,
                            2,
                        ),
                        "is_synthetic_voice": display_is_synthetic,
                        "vad_active": True,
                        "vad_rms": round(chunk_rms, 6),
                        "analysis_window_samples": LIVE_WINDOW_SAMPLES,
                        "analysis_window_seconds": round(
                            LIVE_WINDOW_SAMPLES / TARGET_SR,
                            3,
                        ),
                        "sample_rate": TARGET_SR,
                        "inference_backend": result.inference_backend,
                        "confidence_label": result.confidence_label,
                        "latency_ms": result.latency_ms,
                        "clip_duration_secs": result.clip_duration_secs,
                        "feature_diagnostics": diagnostics,
                    },
                    "manipulation_vectors": {
                        "urgency": 0,
                        "authority": 0,
                        "isolation": 0,
                    },
                }
            )

            print(
                "[✓] LIVE MIC | "
                f"Raw: {latest_raw_score * 100.0:.2f}% | "
                f"EMA: {latest_ema_score * 100.0:.2f}% | "
                f"State: {display_state.upper()}"
            )

    except WebSocketDisconnect:
        logger.info("LIVE INTERCEPT: WebSocket disconnected")
        print("\n[-] LIVE INTERCEPT: WebSocket Disconnected")

    except Exception as exc:
        logger.exception("LIVE INTERCEPT: stream error")
        print(f"[!] Live intercept stream error: {exc}")

    finally:
        active = False
        stream_generation += 1

        # Release all per-connection state.
        audio_buffer = np.empty(0, dtype=np.int16)
        ema_score = None
        confirmed_state = "idle"
        candidate_state = "idle"
        candidate_hits = 0
        latest_raw_score = 0.0
        latest_ema_score = 0.0
        latest_speaker_id = None
        latest_speaker_similarity = 0.0
        speech_active = False
        last_speech_time = None

        print(
            "[-] LIVE INTERCEPT: Per-connection forensic state released."
        )
