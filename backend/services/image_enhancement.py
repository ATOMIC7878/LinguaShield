import cv2
import numpy as np
import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

# Configure high-fidelity local execution logging parameters
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger("LinguaShield.ImageEnhancement")

@dataclass(frozen=True)
class EnhancedImageVariant:
    """Dataclass holding a processed image matrix variant alongside its pipeline metadata."""
    name: str
    matrix: np.ndarray
    description: str
    metadata: dict = field(default_factory=dict)

class ImageEnhancementEngine:
    """Advanced Multi-Stage Preprocessing Engine optimized for isolating 
    dense, non-dictionary alphanumeric strings and micro-punctuation tokens inside URLs.
    """
    
    def __init__(self, sr_model_path: Optional[str] = None):
        self.sr_model_path = sr_model_path
        self.sr_active = False
        self.sr_net = None
        
        # Defensive check for local AI Super Resolution weights files
        if self.sr_model_path and os.path.exists(self.sr_model_path):
            try:
                # Expecting FSRCNN or EDSR models configured for OpenCV
                model_name = os.path.basename(self.sr_model_path).lower()
                if "fsrcnn" in model_name:
                    self.sr_net = cv2.dnn_superres.DnnSuperResImpl_create()
                    self.sr_net.readModel(self.sr_model_path)
                    self.sr_net.setModel("fsrcnn", 3)
                    self.sr_active = True
                elif "edsr" in model_name:
                    self.sr_net = cv2.dnn_superres.DnnSuperResImpl_create()
                    self.sr_net.readModel(self.sr_model_path)
                    self.sr_net.setModel("edsr", 3)
                    self.sr_active = True
                logger.info(f"[✓] Neural Super Resolution engine initialized via: {model_name}")
            except Exception as e:
                logger.warning(f"[!] SR Weights found but failed to compile: {e}. Falling back to spatial interpolation.")

    def process_source_bytes(self, file_bytes: bytes) -> List[EnhancedImageVariant]:
        """Ingests raw file upload binary buffers, maps them into matrix arrays, 
        and routes them through the multi-stage optimization framework.
        """
        try:
            # Reconstruct binary stream into basic OpenCV BGR image matrix array
            nparr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("Inbound binary buffer could not be decoded into a valid image matrix map.")
                
            return self.generate_all_variants(img)
            
        except Exception as error:
            logger.error(f"Fatal disruption during file byte decoding pass: {str(error)}")
            raise

    def generate_all_variants(self, source_img: np.ndarray) -> List[EnhancedImageVariant]:
        """Orchestrates the image transform variants, generating an ensemble collection 
        of enhanced image matrices.
        """
        variants: List[EnhancedImageVariant] = []
        
        # 1. Base Normalization: Map source input to standard Grayscale matrix
        gray = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
        
        # 2. Universal Spatial Scaling / Resolution Expansion Pass
        # Enforces a baseline 3x resolution increase to expand character pixel footprint boundaries.
        if self.sr_active and self.sr_net:
            try:
                base_scaled = self.sr_net.upsample(gray)
                logger.info("[Preprocessor] Image upscaled using Neural Super-Resolution model.")
            except Exception as sr_err:
                logger.warning(f"[!] Live SR Upscale fault: {sr_err}. Reverting to Lanczos4/Cubic grid.")
                height, width = gray.shape[:2]
                base_scaled = cv2.resize(gray, (int(width * 3.0), int(height * 3.0)), interpolation=cv2.INTER_LANCZOS4)
        else:
            height, width = gray.shape[:2]
            base_scaled = cv2.resize(gray, (int(width * 3.0), int(height * 3.0)), interpolation=cv2.INTER_CUBIC)
            
        # Add Preprocessed Base Scaled Image to collection list
        variants.append(EnhancedImageVariant(
            name="base_gray_scaled",
            matrix=base_scaled,
            description="3x Spatial Interpolated Grayscale Base Layer"
        ))
        
        # ---------------------------------------------------------------------
        # VARIANT A: Smooth Vector Contrast (Optimized for EasyOCR)
        # ---------------------------------------------------------------------
        try:
            # Apply local CLAHE contrast balancing
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            clahe_applied = clahe.apply(base_scaled)
            
            # Apply Bilateral Filter to smooth noise while keeping character edge contours sharp
            smooth_contrast = cv2.bilateralFilter(clahe_applied, d=9, sigmaColor=75, sigmaSpace=75)
            
            variants.append(EnhancedImageVariant(
                name="variant_smooth_contrast",
                matrix=smooth_contrast,
                description="CLAHE Contrast and Bilateral Edge Smoothing Filter"
            ))
        except Exception as err:
            logger.error(f"Failed to generate Variant A: {err}")

        # ---------------------------------------------------------------------
        # VARIANT B: High-Fidelity Edge Sharpening (Optimized for Faint Punctuation)
        # ---------------------------------------------------------------------
        try:
            # Clear high-frequency screenshot compression noise using Non-Local Means Denoising
            denoised = cv2.fastNlMeansDenoising(base_scaled, None, h=10, templateWindowSize=7, searchWindowSize=21)
            
            # Apply Unsharp Masking to emphasize contrast differences at sharp changes
            blur = cv2.GaussianBlur(denoised, (0, 0), 2.0)
            unsharp_mask = cv2.addWeighted(denoised, 2.5, blur, -1.5, 0)
            
            # Add micro-Laplacian edge highlight maps to sharpen characters
            laplacian_map = cv2.Laplacian(unsharp_mask, cv2.CV_64F, ksize=3)
            laplacian_scaled = cv2.convertScaleAbs(laplacian_map)
            sharp_fusion = cv2.addWeighted(unsharp_mask, 1.0, laplacian_scaled, 0.3, 0)
            
            variants.append(EnhancedImageVariant(
                name="variant_edge_sharpened",
                matrix=sharp_fusion,
                description="Non-Local Means Denoised and Gaussian/Laplacian Unsharp Masking"
            ))
        except Exception as err:
            logger.error(f"Failed to generate Variant B: {err}")

        # ---------------------------------------------------------------------
        # VARIANT C: Adaptive Segmentation Binary (Optimized for Tesseract)
        # ---------------------------------------------------------------------
        try:
            # Generate high-contrast black/white pixels using local localized blocks
            adaptive_bin = cv2.adaptiveThreshold(
                base_scaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 15, 4
            )
            
            # Apply Morphological Closing to weld micro-breaks inside font strokes
            kernel_struct = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            morph_closed = cv2.morphologyEx(adaptive_bin, cv2.MORPH_CLOSE, kernel_struct)
            
            variants.append(EnhancedImageVariant(
                name="variant_adaptive_binary",
                matrix=morph_closed,
                description="Adaptive Gaussian Binarization with Morphological Stroke Closing"
            ))
        except Exception as err:
            logger.error(f"Failed to generate Variant C: {err}")

        # ---------------------------------------------------------------------
        # VARIANT D: Otsu Morphological Matrix (Optimized for Heavy Compressed Artifacts)
        # ---------------------------------------------------------------------
        try:
            # Apply CLAHE first to balance any lighting anomalies
            clahe_pre = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(base_scaled)
            
            # Calculate dynamic global threshold via Otsu integration rules
            _, otsu_bin = cv2.threshold(clahe_pre, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Apply clean morphological Opening to strip standalone noise dots, followed by dynamic dilation
            kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            morph_opened = cv2.morphologyEx(otsu_bin, cv2.MORPH_OPEN, kernel_open)
            
            variants.append(EnhancedImageVariant(
                name="variant_otsu_morph",
                matrix=morph_opened,
                description="Global Otsu Thresholding and Sequence Morphological Opening/Closing Extraction"
            ))
        except Exception as err:
            logger.error(f"Failed to generate Variant D: {err}")

        logger.info(f"[✓] Completed image sweep. Total optimization pipeline variants generated: {len(variants)}")
        return variants