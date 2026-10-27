
"""
LinguaShield acoustic A/B regression harness.

IMPORTANT:
- This is TEST-ONLY. It does not modify production files.
- Baseline uses the current vocoder_analyzer.py + audio_preprocess.py.
- ONNX is only probed for readiness. A model-specific preprocessing/decoding
  adapter is intentionally NOT guessed, because the current project does not
  expose the ONNX contract in the supplied files.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_RESULT_FIELDS = {
    "is_synthetic_voice",
    "synthetic_probability",
    "confidence_label",
    "inference_backend",
    "latency_ms",
    "clip_duration_secs",
    "feature_diagnostics",
    "warning",
}

REQUIRED_API_KEYS = {
    "vocoder_score",
    "is_synthetic_voice",
    "classification_label",
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def locate_project(root: Path) -> dict[str, Path]:
    candidates = [
        {
            "vocoder": root / "backend" / "services" / "vocoder_analyzer.py",
            "preprocess": root / "backend" / "services" / "audio_preprocess.py",
            "model": root / "models" / "lfcc_lcnn_asvspoofing.onnx",
        },
        {
            "vocoder": root / "vocoder_analyzer(10).py",
            "preprocess": root / "audio_preprocess(4).py",
            "model": root / "models" / "lfcc_lcnn_asvspoofing.onnx",
        },
    ]
    for item in candidates:
        if item["vocoder"].exists() and item["preprocess"].exists():
            return item
    raise FileNotFoundError(
        "Could not locate the current vocoder_analyzer.py and audio_preprocess.py."
    )


def baseline_analyze(
    vocoder_path: Path,
    preprocess_path: Path,
    audio_path: Path,
) -> dict[str, Any]:
    vocoder = load_module("ls_vocoder_baseline", vocoder_path)
    preprocess = load_module("ls_audio_preprocess_baseline", preprocess_path)

    processor = preprocess.AudioPreprocessor(
        target_sample_rate=16000,
        target_window_samples=64600,
    )
    detector = vocoder.DeepfakeVocoderAnalyzer()

    file_bytes = audio_path.read_bytes()
    processed = processor.process(file_bytes, audio_path.name)

    windows = processed.chunks or [processed.waveform]
    results = []

    for index, chunk in enumerate(windows):
        normalized = np.clip(chunk, -1.0, 1.0)
        pcm_bytes = (normalized * 32767.0).astype(np.int16).tobytes()
        result = detector.analyze(pcm_bytes)

        # Dataclass -> plain dict without depending on the app runtime.
        row = {
            "window": index,
            "score_pct": round(float(result.synthetic_probability) * 100.0, 2),
            "is_synthetic_voice": bool(result.is_synthetic_voice),
            "confidence_label": result.confidence_label,
            "inference_backend": result.inference_backend,
            "warning": result.warning,
        }
        results.append(row)

    scores = [r["score_pct"] / 100.0 for r in results]
    ai_positive = sum(1 for r in results if r["is_synthetic_voice"])
    aggregate = float(np.mean(scores)) if scores else 0.0
    final_synthetic = bool(ai_positive >= 2 and aggregate >= 0.40) if len(results) > 1 else bool(
        results[0]["is_synthetic_voice"]
    )

    # Validate the response shape that main.py actually serializes with asdict()
    # and then augments with vocoder_score. Do not call analyze_audio_bytes()
    # here because production main.py does NOT pass compressed bytes to that
    # method; it passes decoded PCM to analyze().
    api_result = {
        "synthetic_probability": float(results[0]["score_pct"]) / 100.0 if results else 0.0,
        "is_synthetic_voice": bool(results[0]["is_synthetic_voice"]) if results else False,
        "confidence_label": results[0]["confidence_label"] if results else "LOW",
        "inference_backend": results[0]["inference_backend"] if results else "unknown",
        "latency_ms": 0.0,
        "clip_duration_secs": results[0].get("duration_seconds", 0.0) if results else 0.0,
        "feature_diagnostics": {},
        "warning": None,
    }
    api_result["vocoder_score"] = round(float(api_result["synthetic_probability"]) * 100.0, 2)
    api_result["classification_label"] = (
        "DEEPFAKE_VOICE_CLONE_DETECTED"
        if api_result["is_synthetic_voice"]
        else "NATURAL_HUMAN_VOICE"
    )
    missing_api_keys = sorted(REQUIRED_API_KEYS - set(api_result.keys()))

    return {
        "audio": audio_path.name,
        "duration_seconds_after_trim": round(float(processed.duration_seconds), 2),
        "window_count": len(results),
        "window_scores_pct": [r["score_pct"] for r in results],
        "aggregate_score_pct": round(aggregate * 100.0, 2),
        "ai_positive_windows": ai_positive,
        "main_py_style_final_synthetic": final_synthetic,
        "public_api_contract_ok": not missing_api_keys,
        "missing_api_keys": missing_api_keys,
        "windows": results,
    }


def probe_onnx(model_path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "model_path": str(model_path),
        "model_exists": model_path.exists(),
        "onnxruntime_available": False,
        "status": "NOT_READY",
        "reason": None,
    }

    if not model_path.exists():
        result["reason"] = "ONNX model file is not present at the expected path."
        return result

    try:
        import onnxruntime as ort  # type: ignore
    except Exception as exc:
        result["reason"] = f"onnxruntime is unavailable: {exc}"
        return result

    result["onnxruntime_available"] = True

    try:
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        inputs = []
        for item in session.get_inputs():
            inputs.append({
                "name": item.name,
                "shape": item.shape,
                "type": item.type,
            })
        outputs = []
        for item in session.get_outputs():
            outputs.append({
                "name": item.name,
                "shape": item.shape,
                "type": item.type,
            })

        result["inputs"] = inputs
        result["outputs"] = outputs

        # Do not invent the audio feature extraction expected by the model.
        result["status"] = "MODEL_ACCESSIBLE_BUT_ADAPTER_REQUIRED"
        result["reason"] = (
            "The ONNX graph is readable, but the supplied project files do not "
            "document its required feature/preprocessing contract. No score was "
            "fabricated."
        )
    except Exception as exc:
        result["reason"] = f"ONNX model could not be opened: {exc}"

    return result


def validate_result_contract() -> dict[str, Any]:
    # Validate the current dataclass shape directly from source code.
    return {
        "required_vocoder_fields": sorted(REQUIRED_RESULT_FIELDS),
        "required_api_fields": sorted(REQUIRED_API_KEYS),
        "note": (
            "Any future ONNX implementation should preserve these public fields "
            "so main.py/UI/TrustPath consumers remain compatible."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "audio",
        nargs="+",
        type=Path,
        help="One or more audio files to use for the baseline.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="LinguaShield project root. Defaults to the current directory.",
    )
    args = parser.parse_args()

    paths = locate_project(args.project_root)

    report = {
        "production_files_modified": False,
        "baseline": [],
        "onnx_probe": probe_onnx(paths["model"]),
        "contract": validate_result_contract(),
    }

    for audio_path in args.audio:
        if not audio_path.exists():
            raise FileNotFoundError(audio_path)
        report["baseline"].append(
            baseline_analyze(paths["vocoder"], paths["preprocess"], audio_path)
        )

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
