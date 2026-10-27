# backend/services/audio_preprocess.py

import io
import logging
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass
from pydub import AudioSegment

logger = logging.getLogger("LinguaShield.AudioPreprocess")

@dataclass
class PreprocessedAudioPayload:
    """
    Standardized payload delivered to the Deepfake Voice Detector engine.
    """
    waveform: np.ndarray          # Shape: (N_samples,), dtype: float32, range [-1.0, 1.0]
    sample_rate: int              # Strictly 16000
    original_format: str          # e.g., 'aac', 'mp3', 'wav'
    duration_seconds: float       # Duration after silence trimming
    was_chunked: bool             # True if audio exceeded max_window_samples and was segmented
    chunks: Optional[List[np.ndarray]] = None # List of 64,600-sample windows if long recording


class AudioPreprocessor:
    """
    Enterprise Audio Preprocessing Pipeline for LinguaShield.
    
    Handles:
    - Multi-codec ingestion (.wav, .mp3, .aac, .m4a, .ogg, .opus, .amr)
    - 16kHz Mono PCM conversion
    - Energy-based Voice Activity Detection (VAD) / Silence Trimming
    - Peak amplitude normalization [-1.0, 1.0]
    - Standardized 64,600 sample windowing (~4.03s) required by AASIST/RawNet2
    """
    def __init__(
        self,
        target_sample_rate: int = 16000,
        target_window_samples: int = 64600,  # Exactly ~4.03s for AASIST ASVspoof checkpoints
        silence_threshold_ratio: float = 0.015
    ):
        self.target_sample_rate = target_sample_rate
        self.target_window_samples = target_window_samples
        self.silence_threshold_ratio = silence_threshold_ratio

    def _decode_to_pcm(self, file_bytes: bytes, filename: str) -> Tuple[np.ndarray, str]:
        """
        Decodes binary audio stream to mono 16kHz PCM float32 array.
        """
        file_ext = filename.split(".")[-1].lower() if "." in filename else "wav"
        audio_stream = io.BytesIO(file_bytes)

        try:
            audio_segment = AudioSegment.from_file(audio_stream, format=file_ext)
        except Exception:
            # Fallback for container mismatch
            audio_stream.seek(0)
            audio_segment = AudioSegment.from_file(audio_stream)

        # Ensure 16kHz Mono
        mono_segment = audio_segment.set_frame_rate(self.target_sample_rate).set_channels(1)
        raw_samples = np.array(mono_segment.get_array_of_samples(), dtype=np.float32)

        # Peak normalization to [-1.0, 1.0]
        max_val = np.max(np.abs(raw_samples))
        if max_val > 0:
            raw_samples /= max_val

        return raw_samples, file_ext

    def _trim_silence(self, samples: np.ndarray) -> np.ndarray:
        """
        Energy-based leading and trailing silence removal.
        Prevents synthetic probability dilution from dead call air.
        """
        if len(samples) == 0:
            return samples

        frame_size = int(self.target_sample_rate * 0.02)  # 20ms frames
        hop_size = int(self.target_sample_rate * 0.01)    # 10ms hop
        n_frames = max(1, (len(samples) - frame_size) // hop_size)

        frame_energies = np.array([
            np.sum(samples[i * hop_size : i * hop_size + frame_size] ** 2)
            for i in range(n_frames)
        ])

        max_energy = np.max(frame_energies) + 1e-10
        active_mask = (frame_energies / max_energy) > self.silence_threshold_ratio

        if not np.any(active_mask):
            # If entire audio is quiet, return original array to prevent crash
            return samples

        start_idx = int(np.argmax(active_mask) * hop_size)
        end_idx = int((len(active_mask) - np.argmax(active_mask[::-1]) - 1) * hop_size + frame_size)
        end_idx = min(len(samples), end_idx)

        trimmed = samples[start_idx:end_idx]
        return trimmed if len(trimmed) > self.target_sample_rate * 0.3 else samples

    def _standardize_window(self, samples: np.ndarray) -> Tuple[np.ndarray, bool, List[np.ndarray]]:
        """
        Formats audio into fixed 64,600-sample tensors.
        - If shorter: Tile-pad (repeat audio seamlessly to fill window without silence gaps).
        - If longer: Create non-overlapping chunks for ensemble scoring across the call.
        """
        length = len(samples)

        if length == self.target_window_samples:
            return samples, False, [samples]

        if length < self.target_window_samples:
            # Repeat waveform seamlessly until reaching target window
            repeat_count = (self.target_window_samples // length) + 1
            padded = np.tile(samples, repeat_count)[:self.target_window_samples]
            return padded, False, [padded]

        # Recording exceeds 4 seconds -> Divide into 64,600-sample chunks
        n_chunks = length // self.target_window_samples
        chunks = []
        for i in range(n_chunks):
            start = i * self.target_window_samples
            chunks.append(samples[start : start + self.target_window_samples])

        # Handle remaining trailing segment by padding it
        remainder = length % self.target_window_samples
        if remainder > self.target_sample_rate * 1.0:  # Include trailing audio if > 1s
            trailing = samples[-remainder:]
            repeat_count = (self.target_window_samples // len(trailing)) + 1
            chunks.append(np.tile(trailing, repeat_count)[:self.target_window_samples])

        return chunks[0], True, chunks

    def process(self, file_bytes: bytes, filename: str) -> PreprocessedAudioPayload:
        """
        Executes full preprocessing pipeline and returns structured dataclass payload.
        """
        raw_samples, ext = self._decode_to_pcm(file_bytes, filename)
        trimmed_samples = self._trim_silence(raw_samples)
        duration_sec = len(trimmed_samples) / self.target_sample_rate

        primary_window, was_chunked, chunks = self._standardize_window(trimmed_samples)

        logger.info(
            f"[AudioPreprocess] Ingested '{filename}' ({ext}) | Duration: {duration_sec:.2f}s | "
            f"Chunked: {was_chunked} ({len(chunks)} windows)"
        )

        return PreprocessedAudioPayload(
            waveform=primary_window,
            sample_rate=self.target_sample_rate,
            original_format=ext,
            duration_seconds=round(duration_sec, 2),
            was_chunked=was_chunked,
            chunks=chunks
        )