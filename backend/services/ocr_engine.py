import easyocr
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
import cv2
import numpy as np
import logging
import threading
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure integrated high-fidelity local execution logging parameters
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LinguaShield.OCREngine")

@dataclass(frozen=True)
class OCRBoundingBox:
    """Standardized geometric box model capturing text location matrices."""
    top_left: Tuple[int, int]
    top_right: Tuple[int, int]
    bottom_right: Tuple[int, int]
    bottom_left: Tuple[int, int]

    def to_dict(self) -> Dict[str, List[int]]:
        return {
            "top_left": list(self.top_left),
            "top_right": list(self.top_right),
            "bottom_right": list(self.bottom_right),
            "bottom_left": list(self.bottom_left)
        }

@dataclass(frozen=True)
class OCREngineResult:
    """Unified token tracking container holding granular character extraction outputs."""
    text: str
    confidence: float
    bbox: OCRBoundingBox
    engine_name: str
    variant_origin: str

class MultiEngineOCRCoordinator:
    """Optimized Concurrent Orchestration Layer that runs targeted parallel 
    character sweeps across EasyOCR and Tesseract with strict thread safety.
    """
    
    def __init__(self, tesseract_cmd_path: Optional[str] = None):
        if tesseract_cmd_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd_path
            
        logger.info("[✓] Initializing Thread-Safe Accelerated Multi-Engine OCR Core...")
        self.easyocr_reader = easyocr.Reader(['en', 'hi'], gpu=False)
        self.url_whitelist = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._~:/?#[]@!$&'()*+,;="
        
        # ─── CRITICAL: THREAD SAFETY MUTEX FOR THE DEEP LEARNING MODEL ───
        self.ocr_lock = threading.Lock()

    def run_easyocr_pass(self, img_matrix: np.ndarray, variant_name: str) -> List[OCREngineResult]:
        """Runs deep convolutional character extraction sweeps over targeted image matrices.
        Uses a synchronization lock to prevent cross-thread tensor state corruption.
        """
        results: List[OCREngineResult] = []
        try:
            # Enforce strict sequential execution for the shared neural network instance
            with self.ocr_lock:
                raw_outputs = self.easyocr_reader.readtext(
                    img_matrix,
                    decoder='greedy',
                    paragraph=False,
                    text_threshold=0.20,  # Lowered to capture faint URL components
                    low_text=0.12,        # Retains single-character punctuation maps
                    link_threshold=0.15
                )
            
            for item in raw_outputs:
                bbox_points, text_str, confidence_val = item
                if not text_str.strip():
                    continue
                    
                bbox = OCRBoundingBox(
                    top_left=(int(bbox_points[0][0]), int(bbox_points[0][1])),
                    top_right=(int(bbox_points[1][0]), int(bbox_points[1][1])),
                    bottom_right=(int(bbox_points[2][0]), int(bbox_points[2][1])),
                    bottom_left=(int(bbox_points[3][0]), int(bbox_points[3][1]))
                )
                
                results.append(OCREngineResult(
                    text=text_str.strip(),
                    confidence=float(confidence_val * 100.0),
                    bbox=bbox,
                    engine_name="EasyOCR",
                    variant_origin=variant_name
                ))
        except Exception as err:
            logger.error(f"EasyOCR sweep failed on variant '{variant_name}': {err}")
        return results

    def run_tesseract_pass(self, img_matrix: np.ndarray, variant_name: str, psm: int = 7) -> List[OCREngineResult]:
        """Runs Tesseract character segmentation sweeps stripped of linguistic dictionaries.
        This is completely stateless at the process level and runs perfectly in parallel.
        """
        results: List[OCREngineResult] = []
        try:
            tess_config = (
                f"--oem 1 "
                f"--psm {psm} "
                f"-c load_system_dawg=0 "
                f"-c load_freq_dawg=0 "
                f"-c tessedit_char_whitelist={self.url_whitelist}"
            )
            
            data_frame = pytesseract.image_to_data(
                img_matrix, 
                config=tess_config, 
                output_type=pytesseract.Output.DICT
            )
            
            total_elements = len(data_frame['text'])
            for i in range(total_elements):
                text_str = str(data_frame['text'][i]).strip()
                conf_val = float(data_frame['conf'][i])
                
                if not text_str or conf_val < 15.0:
                    continue
                    
                x, y, w, h = data_frame['left'][i], data_frame['top'][i], data_frame['width'][i], data_frame['height'][i]
                
                bbox = OCRBoundingBox(
                    top_left=(x, y),
                    top_right=(x + w, y),
                    bottom_right=(x + w, y + h),
                    bottom_left=(x, y + h)
                )
                
                results.append(OCREngineResult(
                    text=text_str,
                    confidence=conf_val,
                    engine_name=f"Tesseract_PSM_{psm}",
                    bbox=bbox,
                    variant_origin=variant_name
                ))
        except Exception as err:
            logger.error(f"Tesseract sweep failed on variant '{variant_name}' via PSM {psm}: {err}")
        return results

    def execute_all_engines(self, variant_models: List[Any]) -> List[OCREngineResult]:
        """Coordinates the multi-engine loop concurrently, distributing Tesseract tasks 
        across threads while keeping the deep-learning model passes safe from contention.
        """
        master_raw_collection: List[OCREngineResult] = []
        futures = []

        with ThreadPoolExecutor() as executor:
            for variant in variant_models:
                name = variant.name
                
                # ─── ACCELERATED SMART ROUTING ───
                # 1. EasyOCR processes smooth contrast and base filters sequentially via the lock
                if name in ["base_gray_scaled", "variant_smooth_contrast"]:
                    futures.append(executor.submit(self.run_easyocr_pass, variant.matrix, name))
                
                # 2. Tesseract tackles high-contrast binary frames concurrently across separate cores
                if name in ["variant_adaptive_binary", "variant_otsu_morph"]:
                    futures.append(executor.submit(self.run_tesseract_pass, variant.matrix, name, 7))
                    futures.append(executor.submit(self.run_tesseract_pass, variant.matrix, name, 11))

            # Safely compile results as the asynchronous workers return data arrays
            for future in as_completed(futures):
                master_raw_collection.extend(future.result())

        logger.info(f"[✓] Thread-Safe Multi-Engine Sweep Complete. Collected {len(master_raw_collection)} tokens.")
        return master_raw_collection