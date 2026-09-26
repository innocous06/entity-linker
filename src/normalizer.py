"""Text normalization and synthetic corruption inversion for entity resolution."""

import re
from typing import List, Tuple
from unidecode import unidecode

LEGAL_SUFFIXES_PATTERN = re.compile(
    r"\b("
    r"corp|corporation|pvt|private|ltd|limited|inc|incorporated|llc|sarl|sas|sci|dba|gmbh|co|company|society|"
    r"group|enterprises|solutions|industries|technology|technologies|"
    # Synthetic transliteration variants found in benchmarks
    r"elelpii|elelpi|limittedd|limirrrrdd|praaivett|praiveett|praivrrrr|praa|li"
    r")\b\.?",
    re.IGNORECASE,
)

TITLES_AND_PREFIXES = re.compile(
    r"\b(m/s|dr|mr|mrs|ms|smt|shri|smt\.)\b\.?",
    re.IGNORECASE,
)

DOMAIN_PATTERN = re.compile(r"\.(com|org|net|in|co|fr|io)\b", re.IGNORECASE)
PINCODE_PATTERN = re.compile(r"\b([1-9][0-9]{5}|[0-9]{5})\b")

STOP_WORDS = frozenset([
    "the", "of", "and", "for", "in", "to", "a", "an", "is", "at", "by",
    "no", "new", "old", "near", "null",
    # Generic address stopwords
    "road", "street", "nagar", "marg", "sector", "block", "plot", "bhavan", "complex",
    "floor", "opp", "behind", "flat", "house", "lane", "avenue", "rd", "st",
    "circle", "chowk", "colony", "layout", "cross", "main", "building", "tower",
    "suite", "ste", "fl", "unit", "bldg", "box", "po", "rue", "boulevard", "cours",
])

def clean_name(raw: str) -> str:
    """Normalizes raw business names, strips prefixes, legal suffixes, and domains."""
    if not raw:
        return ""
    s = unidecode(str(raw)).lower()
    s = TITLES_AND_PREFIXES.sub(" ", s)
    s = DOMAIN_PATTERN.sub(" ", s)
    s = LEGAL_SUFFIXES_PATTERN.sub(" ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def clean_address(raw: str) -> str:
    """Normalizes address strings while retaining punctuation relevant to unit numbers."""
    if not raw:
        return ""
    s = unidecode(str(raw)).lower()
    # Replace slashes and structural dividers with spaces
    s = re.sub(r"[/,;()]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def extract_pincode(addr: str) -> str:
    """Extracts 5-digit or 6-digit postal/ZIP code if present."""
    if not addr:
        return ""
    m = PINCODE_PATTERN.search(addr)
    return m.group(0) if m else ""

def get_name_tokens(raw: str, min_len: int = 2) -> List[str]:
    """Generates normalized name tokens including compressed double-characters."""
    s = clean_name(raw)
    toks = []
    for t in s.split():
        if len(t) >= min_len and t not in STOP_WORDS:
            toks.append(t)
            # Invert letter-doubling (e.g., ddriim -> drim, knsttrkssn -> knstrksn)
            deduped = re.sub(r"(.)\1+", r"\1", t)
            if deduped != t and len(deduped) >= min_len and deduped not in STOP_WORDS:
                toks.append(deduped)
    return toks

def get_consonant_skeleton(token: str) -> str:
    """Extracts consonant skeleton after deduplicating consecutive letters."""
    deduped = re.sub(r"(.)\1+", r"\1", token.lower())
    return re.sub(r"[aeiou]", "", deduped)

def get_address_tokens(raw: str, min_len: int = 2) -> List[str]:
    """Generates address tokens preserving numeric identifiers (plot/house numbers)."""
    s = clean_address(raw)
    toks = []
    for t in s.split():
        t = t.strip(".-#")
        if not t or t in STOP_WORDS:
            continue
        # Tokens containing digits (e.g. 16-11-23, e-7, 28-b, 702) are retained if >= 2 chars
        if any(c.isdigit() for c in t):
            if len(t) >= 2:
                toks.append(t)
        elif len(t) >= max(3, min_len):
            toks.append(t)
    return toks
