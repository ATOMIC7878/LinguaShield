import io
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
from pydub import AudioSegment
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.utils.fetching import LocalStrategy

logger = logging.getLogger("LinguaShield.VoiceprintVerifier")


class VoiceprintVerifier:
    """
    Biometric Voiceprint Verification Engine for LinguaShield.

    Uses SpeechBrain's pretrained ECAPA-TDNN speaker encoder to generate a
    learned 192-dimensional speaker embedding, then performs cosine-similarity
    verification against the enrolled target profile.

    The public API and return fields are intentionally unchanged.
    """

    def __init__(self, sample_rate: int = 16000, similarity_threshold: float = 0.88):
        self.sample_rate = sample_rate
        self.similarity_threshold = similarity_threshold
        self._enrolled_profiles: Dict[str, np.ndarray] = {}
        self._speaker_encoder: Optional[EncoderClassifier] = None
        self._model_dir = (
            Path(__file__).resolve().parents[1]
            / "models"
            / "speaker_encoder"
            / "spkrec-ecapa-voxceleb"
        )

    def _get_speaker_encoder(self) -> EncoderClassifier:
        """Load the locally downloaded ECAPA-TDNN model once, on demand."""
        if self._speaker_encoder is not None:
            return self._speaker_encoder

        if not self._model_dir.exists():
            raise FileNotFoundError(
                f"ECAPA-TDNN model directory not found: {self._model_dir}"
            )

        logger.info(
            f"[Voiceprint] Loading local ECAPA-TDNN speaker encoder from: {self._model_dir}"
        )

        self._speaker_encoder = EncoderClassifier.from_hparams(
            source=str(self._model_dir),
            savedir=str(self._model_dir),
            run_opts={"device": "cpu"},
            local_strategy=LocalStrategy.COPY,
        )

        logger.info("[Voiceprint] ECAPA-TDNN speaker encoder loaded successfully.")
        return self._speaker_encoder

    @staticmethod
    def _normalize_embedding(embedding_vector: np.ndarray) -> np.ndarray:
        """L2-normalize a speaker embedding without altering its dimensions."""
        embedding_vector = np.asarray(embedding_vector, dtype=np.float32).reshape(-1)
        if embedding_vector.size == 0:
            raise ValueError("embedding_vector cannot be empty")

        norm = np.linalg.norm(embedding_vector)
        if norm <= 1e-10:
            raise ValueError("embedding_vector has zero magnitude")

        return (embedding_vector / norm).astype(np.float32)

    def register_executive_profile(self, target_id: str, embedding_vector: np.ndarray) -> None:
        """Enroll or update an executive speaker profile vector."""
        if not isinstance(target_id, str) or not target_id.strip():
            raise ValueError("target_id cannot be blank")

        normalized_vec = self._normalize_embedding(embedding_vector)
        self._enrolled_profiles[target_id.strip().lower()] = normalized_vec
        logger.info(
            f"[Voiceprint Registry] Enrolled biometric profile for target ID: '{target_id}'"
        )

    def _extract_speaker_embedding(self, pcm_data: np.ndarray) -> np.ndarray:
        """Extract a learned 192-dimensional ECAPA-TDNN speaker embedding."""
        samples = np.asarray(pcm_data, dtype=np.float32).reshape(-1)

        if samples.size < int(self.sample_rate * 0.5):
            return np.zeros(192, dtype=np.float32)

        # Match the expected waveform convention: mono, 16 kHz, floating point
        # samples in approximately [-1, 1]. The decoder already resamples to
        # self.sample_rate, but normalization here also covers the enrollment API
        # which supplies raw AudioSegment samples directly.
        samples = samples - np.mean(samples)
        peak = float(np.max(np.abs(samples)))
        if peak > 1e-10:
            samples = samples / peak

        waveform = torch.from_numpy(samples.astype(np.float32)).unsqueeze(0)
        encoder = self._get_speaker_encoder()

        with torch.inference_mode():
            embedding = encoder.encode_batch(waveform)

        embedding = embedding.detach().cpu().numpy().reshape(-1).astype(np.float32)

        if embedding.size != 192:
            raise RuntimeError(
                f"Unexpected ECAPA embedding size: {embedding.size}; expected 192"
            )

        return self._normalize_embedding(embedding)

    @staticmethod
    def _compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Compute normalized cosine similarity in [0, 1]."""
        a = np.asarray(vec_a, dtype=np.float32).reshape(-1)
        b = np.asarray(vec_b, dtype=np.float32).reshape(-1)

        if a.size == 0 or b.size == 0:
            return 0.0

        if a.size != b.size:
            return 0.0

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a <= 1e-10 or norm_b <= 1e-10:
            return 0.0

        cosine = float(np.dot(a, b) / (norm_a * norm_b))
        return float(np.clip(cosine, 0.0, 1.0))

    def _decode_audio(self, file_bytes: bytes, filename: str) -> np.ndarray:
        """Decode common containers to normalized 16 kHz mono PCM."""
        file_extension = filename.split(".")[-1].lower() if "." in filename else "wav"
        audio_stream = io.BytesIO(file_bytes)

        try:
            audio_segment = AudioSegment.from_file(audio_stream, format=file_extension)
        except Exception:
            audio_stream.seek(0)
            audio_segment = AudioSegment.from_file(audio_stream)

        mono_audio = audio_segment.set_frame_rate(self.sample_rate).set_channels(1)
        raw_pcm = np.asarray(mono_audio.get_array_of_samples(), dtype=np.float32)
        if raw_pcm.size == 0:
            return raw_pcm

        peak = float(np.max(np.abs(raw_pcm)))
        if peak > 0:
            raw_pcm /= peak
        return raw_pcm

    def verify_audio_bytes(
        self,
        file_bytes: bytes,
        filename: str,
        claimed_target_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Extract a speaker embedding and compare it with the enrolled target."""
        try:
            if not claimed_target_id:
                return {
                    "biometric_verified": True,
                    "similarity_score": 100.0,
                    "claimed_identity": None,
                    "impersonation_risk": 0.0,
                    "classification_label": "TARGET_NOT_SPECIFIED_VERIFICATION_SKIPPED",
                }

            target_key = str(claimed_target_id).strip().lower()
            if not target_key:
                return {
                    "biometric_verified": True,
                    "similarity_score": 100.0,
                    "claimed_identity": None,
                    "impersonation_risk": 0.0,
                    "classification_label": "TARGET_NOT_SPECIFIED_VERIFICATION_SKIPPED",
                }

            if target_key not in self._enrolled_profiles:
                logger.warning(
                    f"[Voiceprint] Claimed target '{claimed_target_id}' not found in registry."
                )
                return {
                    "biometric_verified": False,
                    "similarity_score": 0.0,
                    "claimed_identity": claimed_target_id,
                    "impersonation_risk": 85.0,
                    "classification_label": "UNENROLLED_TARGET_IMPERSONATION_SUSPECTED",
                }

            raw_pcm = self._decode_audio(file_bytes, filename)
            if raw_pcm.size == 0:
                return {
                    "biometric_verified": False,
                    "similarity_score": 0.0,
                    "claimed_identity": claimed_target_id,
                    "impersonation_risk": 0.0,
                    "classification_label": "AUDIO_BUFFER_EMPTY",
                }

            live_embedding = self._extract_speaker_embedding(raw_pcm)
            enrolled_embedding = self._enrolled_profiles[target_key]

            # A zero vector indicates unusable/too-short audio and must never
            # become a successful biometric match.
            if np.linalg.norm(live_embedding) <= 1e-10:
                return {
                    "biometric_verified": False,
                    "similarity_score": 0.0,
                    "claimed_identity": claimed_target_id,
                    "impersonation_risk": 100.0,
                    "classification_label": "VOICEPRINT_AUDIO_INSUFFICIENT",
                }

            similarity = self._compute_cosine_similarity(
                live_embedding, enrolled_embedding
            )
            similarity_percent = round(similarity * 100.0, 2)

            # Keep the existing threshold for now; it will be recalibrated after
            # ECAPA-based V5/V6/V7 verification testing.
            is_verified = similarity >= self.similarity_threshold
            impersonation_risk = round(max(0.0, 1.0 - similarity) * 100.0, 2)

            logger.info(
                f"[Voiceprint] Target: '{claimed_target_id}' | "
                f"ECAPA Similarity Score: {similarity_percent}% | Verified: {is_verified}"
            )

            return {
                "biometric_verified": is_verified,
                "similarity_score": similarity_percent,
                "claimed_identity": claimed_target_id,
                "impersonation_risk": impersonation_risk,
                "classification_label": (
                    "BIOMETRIC_VOICEPRINT_MATCHED"
                    if is_verified
                    else "CEO_IMPERSONATION_VOICE_MISMATCH"
                ),
            }

        except Exception as e:
            logger.error(f"[Voiceprint] Verification failed: {e}")
            return {
                "biometric_verified": False,
                "similarity_score": 0.0,
                "claimed_identity": claimed_target_id,
                "impersonation_risk": 0.0,
                "classification_label": "VOICEPRINT_VERIFICATION_ERROR",
            }
