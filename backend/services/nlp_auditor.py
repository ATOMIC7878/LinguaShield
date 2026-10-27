import re

def calculate_psycholinguistic_vectors(text: str):
    """Parses raw text strings to calculate intensity scores (0-100) 
    across our three core social engineering manipulation axes.
    """
    text_lower = text.lower()
    
    # Vocabulary mappings representing core systemic threats
    urgency_words = ["immediate", "now", "urgent", "hours", "minutes", "final notice", "disconnect", "expired", "asap", "limited"]
    authority_words = ["cbi", "police", "court", "arrest", "customs", "officer", "bank", "it desk", "compliance", "verification"]
    isolation_words = ["confidential", "secret", "dont tell", "eyes and ears", "keep to yourself", "don't alert", "stay online"]
    
    # Frequency parsing matrices
    urg_score = sum(1 for word in urgency_words if word in text_lower)
    aut_score = sum(1 for word in authority_words if word in text_lower)
    iso_score = sum(1 for word in isolation_words if word in text_lower)
    
    # structural panic offsets
    punctuation_bonus = min(text.count('!') + text.count('?'), 5) * 10
    
    # Data scaling normalization
    urgency_percentage = min((urg_score * 25) + punctuation_bonus, 100)
    authority_percentage = min(aut_score * 30, 100)
    isolation_percentage = min(iso_score * 40, 100)
    
    return {
        "urgency": int(urgency_percentage),
        "authority": int(authority_percentage),
        "isolation": int(isolation_percentage)
    }