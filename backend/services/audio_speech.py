import io
import os
import re
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import speech_recognition as sr
from pydub import AudioSegment
from pydub.effects import normalize, compress_dynamic_range

# Configure highly accurate forensic system tracking logs
logger = logging.getLogger("LinguaShield.AudioProcessor")

# Universal regex profiles for dynamic script separation
LATIN_PATTERN = re.compile(r'[a-zA-Z0-9\.\@\-\_]+')
DIGIT_OR_SYMBOL_PATTERN = re.compile(r'[@\.]|\d+')

def apply_telephony_acoustic_conditioning(audio_segment: AudioSegment) -> AudioSegment:
    """
    Applies strict bandpass filtering and dynamic range compression to isolate 
    vocal frequencies from heavy background noise (fans, traffic, TV static).
    """
    logger.info("[Acoustic Conditioning] Running bandpass filter (300Hz - 3400Hz)...")
    
    # 1. Apply high-pass filter to eliminate low-end rumbles (fans, traffic)
    conditioned_audio = audio_segment.high_pass_filter(300)
    
    # 2. Apply low-pass filter to clean high-frequency system hiss
    conditioned_audio = conditioned_audio.low_pass_filter(3400)
    
    # 3. Dynamic Range Companding: Level out high-volume pops and faint speech whispers
    logger.info("[Acoustic Conditioning] Executing dynamic range companding...")
    conditioned_audio = compress_dynamic_range(
        conditioned_audio,
        threshold=-24.0,
        ratio=4.0,
        attack=5.0,
        release=50.0
    )
    
    return normalize(conditioned_audio)

def process_dynamic_multilingual_alignment(audio_data: sr.AudioData, recognizer: sr.Recognizer, languages: list) -> dict:
    """
    Executes simultaneous parallel transcription passes for ANY dynamic list of target languages.
    Returns a dictionary dividing the dominant primary transcript from supplementary script data.
    """
    raw_transcripts = {}
    
    with ThreadPoolExecutor(max_workers=len(languages)) as executor:
        future_to_lang = {
            executor.submit(recognizer.recognize_google, audio_data, language=lang): lang 
            for lang in languages
        }
        
        for future in as_completed(future_to_lang):
            lang = future_to_lang[future]
            try:
                result = future.result().strip()
                if result:
                    raw_transcripts[lang] = result
            except Exception:
                raw_transcripts[lang] = ""

    if not any(raw_transcripts.values()):
        return {"primary": "", "secondary": ""}
        
    # Dynamically select the primary anchor language transcript based on highest token density
    sorted_transcripts = sorted(raw_transcripts.items(), key=lambda item: len(item[1].split()), reverse=True)
    primary_lang, base_transcript = sorted_transcripts[0]
    
    logger.info(f"[Linguistic Core] Dominant Engine Pass Locked: [{primary_lang}]")
    
    unified_text_pool = base_transcript.lower()
    supplementary_tokens = []
    
    # ─── DYNAMIC SCRIPT CROSS-ALIGNMENT INTERLEAVER ───
    for lang, text in raw_transcripts.items():
        if lang == primary_lang or not text:
            continue
            
        words = text.split()
        for word in words:
            clean_word = word.lower().strip()
            
            if clean_word in unified_text_pool:
                continue
                
            if LATIN_PATTERN.match(word):
                if DIGIT_OR_SYMBOL_PATTERN.search(clean_word):
                    if word not in supplementary_tokens:
                        supplementary_tokens.append(word)
                        unified_text_pool += f" {clean_word}"
                elif len(clean_word) > 2 and word not in supplementary_tokens:
                    supplementary_tokens.append(word)
                    unified_text_pool += f" {clean_word}"
            else:
                if clean_word not in unified_text_pool and word not in supplementary_tokens:
                    supplementary_tokens.append(word)
                    unified_text_pool += f" {clean_word}"
                    
    return {
        "primary": base_transcript.strip(),
        "secondary": " ".join(supplementary_tokens).strip()
    }

def merge_overlapping_transcripts(chunks_text_list: list) -> str:
    """
    Production-grade overlap reconstruction engine. Merges sliding window transcripts 
    and introduces a dynamic fuzzy prefix alignment layer to eliminate seam phrase mirroring.
    """
    if not chunks_text_list:
        return ""
        
    unified_words = []
    
    for block_text in chunks_text_list:
        if not block_text.strip():
            continue
            
        current_words = block_text.split()
        if not unified_words:
            unified_words.extend(current_words)
            continue
            
        # 1. Structural Sequence Match Pass
        max_overlap_check = min(15, len(current_words), len(unified_words))
        actual_overlap_count = 0
        
        unified_lower = [w.lower() for w in unified_words]
        current_lower = [w.lower() for w in current_words]
        
        for match_size in range(1, max_overlap_check + 1):
            if unified_lower[-match_size:] == current_lower[:match_size]:
                actual_overlap_count = match_size
                
        # 2. Fuzzy Prefix Alignment Pass
        if actual_overlap_count == 0:
            for match_size in range(max_overlap_check, 2, -1):
                if unified_lower[-match_size:-1] == current_lower[:match_size-1]:
                    actual_overlap_count = match_size - 1
                    logger.info(f"[Sequence Alignment] Resolved fuzzy phrase boundary overlap shift at {match_size} tokens.")
                    break
                
        # Trim sequential matching boundaries safely
        words_to_add = current_words[actual_overlap_count:]
        
        # 3. Localized Seam-Alignment Deduplicator
        if unified_words and words_to_add:
            tail_check_zone = [w.lower() for w in unified_words[-4:]]
            cleaned_words_to_add = []
            for w in words_to_add:
                if w.lower() in tail_check_zone and (DIGIT_OR_SYMBOL_PATTERN.search(w.lower()) or len(w) > 4):
                    if w.lower() not in [cw.lower() for cw in cleaned_words_to_add]:
                        continue
                cleaned_words_to_add.append(w)
            words_to_add = cleaned_words_to_add
                
        unified_words.extend(words_to_add)
        
    return " ".join(unified_words)

def transcribe_audio_to_text(file_bytes: bytes, filename: str, target_lang: str = "hi-IN") -> str:
    """
    Ingests binary audio/video arrays, applies telephony noise conditioning, 
    and runs a dynamic multilingual sliding window engine across explicit target language scopes.
    Formats transcripts cleanly by isolating structural sentences from metadata token lists.
    """
    logger.info(f"--- Launching Dynamic Multilingual Forensic Ingestion Pipeline for: {filename} ---")
    
    PROTOTYPE_TARGET_LANGUAGES = ["hi-IN", "en-IN", "mr-IN"]
    
    try:
        file_extension = filename.split(".")[-1].lower()
        if file_extension in ['mid', 'midi']:
            return "ERROR: MIDI instruction data arrays contain no vocal sound tracks."
            
        audio_stream = io.BytesIO(file_bytes)
        try:
            audio_segment = AudioSegment.from_file(audio_stream, format=file_extension)
        except Exception:
            audio_stream.seek(0)
            audio_segment = AudioSegment.from_file(audio_stream)
            
        conditioned_audio = apply_telephony_acoustic_conditioning(audio_segment)
        conditioned_audio = conditioned_audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        
        silence_pad = AudioSegment.silent(duration=500, frame_rate=16000).set_channels(1).set_sample_width(2)
        conditioned_audio = silence_pad + conditioned_audio
        
        total_duration_ms = len(conditioned_audio)
        duration_seconds = total_duration_ms / 1000.0
        logger.info(f"[Forensic Tracker] Conditioned Audio Timeline: {duration_seconds:.2f} seconds.")
        
        window_duration_ms = 15000 
        overlap_margin_ms = 4000   
        
        primary_windows = []
        secondary_windows = []
        
        start_pointer = 0
        window_index = 1
        
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 25  
        recognizer.dynamic_energy_threshold = False
        
        logger.info("[Forensic Tracker] Commencing parallel sliding window text extraction sweeps...")
        
        while start_pointer < total_duration_ms:
            end_pointer = min(start_pointer + window_duration_ms, total_duration_ms)
            logger.info(f"[Processing Window #{window_index}] Enforcing bounds: {start_pointer/1000.0}s to {end_pointer/1000.0}s")
            
            chunk_segment = conditioned_audio[start_pointer:end_pointer]
            
            chunk_wav_buffer = io.BytesIO()
            chunk_segment.export(chunk_wav_buffer, format="wav")
            chunk_wav_buffer.seek(0)
            
            try:
                with sr.AudioFile(chunk_wav_buffer) as chunk_source:
                    audio_data = recognizer.record(chunk_source)
                
                alignment_matrix = process_dynamic_multilingual_alignment(
                    audio_data, recognizer, PROTOTYPE_TARGET_LANGUAGES
                )
                
                if alignment_matrix["primary"].strip():
                    primary_windows.append(alignment_matrix["primary"].strip())
                if alignment_matrix["secondary"].strip():
                    secondary_windows.append(alignment_matrix["secondary"].strip())
                    
            except Exception as chunk_fault:
                logger.error(f"[Processing Window #{window_index}] Sweeper encountered a validation tracking anomaly: {chunk_fault}")
                
            start_pointer += (window_duration_ms - overlap_margin_ms)
            window_index += 1
            
            if total_duration_ms - start_pointer < 1000:
                break
                
        # Reconstruct and merge overlapping transcripts completely independently
        logger.info("[Forensic Tracker] Assembling segmented text blocks via sequence alignment engines...")
        final_primary = merge_overlapping_transcripts(primary_windows)
        final_secondary = merge_overlapping_transcripts(secondary_windows)
        
        # Standardize text formatting
        final_primary = re.sub(r'\s+', ' ', final_primary).strip()
        
        # 🌟 PRESENTATION RE-DESIGN: Convert space-separated fragments into a clean comma list of unique tokens
        if final_secondary.strip():
            raw_tokens = final_secondary.split()
            # Order-preserving deduplication using dict keys
            unique_tokens = list(dict.fromkeys(raw_tokens))
            formatted_tokens = ", ".join(unique_tokens)
            
            final_complete_transcript = f"{final_primary}\n\n[Cross-Script Ingestion Tokens]\n{formatted_tokens}"
        else:
            final_complete_transcript = final_primary
            
        logger.info(f"[✓ Production Matrix Complete] Fully Transcribed Character Length: {len(final_complete_transcript)}")
        return final_complete_transcript if final_complete_transcript.strip() else "ERROR: Audio processed successfully but contained no identifiable speech tokens."
        
    except Exception as hardware_pipeline_fault:
        logger.critical(f"[!] Critical core pipeline crash: {hardware_pipeline_fault}")
        return f"ERROR: Processing engine failed to resolve media maps. Details: {str(hardware_pipeline_fault)}"