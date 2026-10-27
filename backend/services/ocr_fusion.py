import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
from services.ocr_engine import OCREngineResult, OCRBoundingBox

# Configure high-fidelity local execution logging parameters
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LinguaShield.OCRFusion")

@dataclass
class FusedTextToken:
    """Unified structural data token representing text aggregated across spatial matrices."""
    text: str
    confidence: float
    bbox: OCRBoundingBox
    source_engines: List[str] = field(default_factory=list)
    source_variants: List[str] = field(default_factory=list)

class OCRFusionEngine:
    """Advanced Spatial Layout Engine that clusters, sorts, de-duplicates, 
    and welds fragmented OCR multi-engine outputs into structurally accurate tokens.
    """
    
    def __init__(self, vertical_line_threshold: int = 25, horizontal_weld_threshold: int = 30):
        # Thresholds tuned for the 3x scaled image matrix grid generated in Module 1
        self.y_threshold = vertical_line_threshold
        self.x_threshold = horizontal_weld_threshold

    def _get_bbox_center_y(self, bbox: OCRBoundingBox) -> int:
        """Calculates the vertical midpoint center of a bounding coordinate box."""
        return int((bbox.top_left[1] + bbox.bottom_left[1]) / 2)

    def cluster_into_horizontal_lines(self, tokens: List[OCREngineResult]) -> List[List[OCREngineResult]]:
        """Groups raw spatial token collections into distinct horizontal line rows 

        based on vertical center alignments.
        """
        if not tokens:
            return []

        # Sort all tokens by top-left Y coordinate initially to group row-by-row
        sorted_by_y = sorted(tokens, key=lambda t: t.bbox.top_left[1])
        lines: List[List[OCREngineResult]] = []

        for token in sorted_by_y:
            token_center_y = self._get_bbox_center_y(token.bbox)
            assigned_to_line = False

            # Check if this token belongs to an existing horizontal line track
            for line in lines:
                line_center_y = sum(self._get_bbox_center_y(t.bbox) for t in line) / len(line)
                if abs(token_center_y - line_center_y) <= self.y_threshold:
                    line.append(token)
                    assigned_to_line = True
                    break

            # If it doesn't align with current rows, establish a new horizontal line bucket
            if not assigned_to_line:
                lines.append([token])

        return lines

    def fuse_and_weld_tokens(self, raw_tokens: List[OCREngineResult]) -> List[FusedTextToken]:
        """Executes the complete spatial fusion loop: bucket rows, sort horizontally, 

        resolve engine duplicates, and weld fragmented word/URL spaces.
        """
        if not raw_tokens:
            return []

        # Step 1: Cluster items into physical horizontal rows
        horizontal_lines = self.cluster_into_horizontal_lines(raw_tokens)
        final_fused_tokens: List[FusedTextToken] = []

        for row_index, line in enumerate(horizontal_lines):
            # Step 2: Sort the line internally from left-to-right (X axis coordinates)
            sorted_line = sorted(line, key=lambda t: t.bbox.top_left[0])
            
            if not sorted_line:
                continue

            # Step 3: De-duplicate overlapping engine bounding hits (Spatial NMS)
            resolved_line_tokens: List[FusedTextToken] = []
            for token in sorted_line:
                # Wrap internal structures inside the unified FusedTextToken container
                current_fused = FusedTextToken(
                    text=token.text,
                    confidence=token.confidence,
                    bbox=token.bbox,
                    source_engines=[token.engine_name],
                    source_variants=[token.variant_origin]
                )

                if not resolved_line_tokens:
                    resolved_line_tokens.append(current_fused)
                    continue

                last_resolved = resolved_line_tokens[-1]
                
                # Check for spatial intersection overlap overlap (horizontal bounding hit collision)
                # If they start at a similar point, they are duplicates from separate engines/variants
                if abs(token.bbox.top_left[0] - last_resolved.bbox.top_left[0]) < 20:
                    # Keep the token that carries the highest confidence parameters
                    if current_fused.confidence > last_resolved.confidence:
                        resolved_line_tokens[-1] = current_fused
                    else:
                        # Append source trace histories to track lineage for debugging/logging
                        if current_fused.source_engines[0] not in last_resolved.source_engines:
                            last_resolved.source_engines.extend(current_fused.source_engines)
                        if current_fused.source_variants[0] not in last_resolved.source_variants:
                            last_resolved.source_variants.extend(current_fused.source_variants)
                else:
                    resolved_line_tokens.append(current_fused)

            # Step 4: Proximity Character Welding (Fixes split URL segments)
            welded_line_tokens: List[FusedTextToken] = []
            for token in resolved_line_tokens:
                if not welded_line_tokens:
                    welded_line_tokens.append(token)
                    continue

                last_welded = welded_line_tokens[-1]
                
                # Calculate horizontal gap between last item right edge and current item left edge
                horizontal_gap = token.bbox.top_left[0] - last_welded.bbox.top_right[0]

                # If distance is small, weld the text fragments together without spaces
                if 0 <= horizontal_gap <= self.x_threshold:
                    # Handle structural punctuation stitching securely (e.g., 'www', '.', 'google')
                    last_welded.text += token.text
                    
                    # Update geometric box bounds model to stretch across the newly welded layout area
                    last_welded.bbox = OCRBoundingBox(
                        top_left=last_welded.bbox.top_left,
                        top_right=token.bbox.top_right,
                        bottom_right=token.bbox.bottom_right,
                        bottom_left=last_welded.bbox.bottom_left
                    )
                    # Average the confidence values across the welded segments
                    last_welded.confidence = (last_welded.confidence + token.confidence) / 2.0
                    last_welded.source_engines.extend([e for e in token.source_engines if e not in last_welded.source_engines])
                    last_welded.source_variants.extend([v for v in token.source_variants if v not in last_welded.source_variants])
                else:
                    welded_line_tokens.append(token)

            final_fused_tokens.extend(welded_line_tokens)

        logger.info(f"[✓] Spatial Fusion Step Complete. Staged {len(final_fused_tokens)} structurally sorted word blocks.")
        return final_fused_tokens

    def compile_to_master_string(self, fused_tokens: List[FusedTextToken]) -> str:
        """Converts the sorted token blocks into a single space-separated string 

        suitable for text classification models.
        """
        return " ".join([token.text for token in fused_tokens])