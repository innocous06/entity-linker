"""Feature extraction pipeline for candidate entity pairs."""

import re
from typing import Dict, List, Tuple
import jellyfish
import numpy as np
import polars as pl
from Levenshtein import ratio as lev_ratio

from src.normalizer import clean_name, clean_address, extract_pincode, get_consonant_skeleton

FEATURE_NAMES = [
    "jaro_winkler_name",
    "levenshtein_ratio_name",
    "token_sort_ratio_name",
    "token_set_ratio_name",
    "name_len_diff",
    "name_len_ratio",
    "exact_name_match",
    "first_token_match",
    "soundex_match_flag",
    "metaphone_match_flag",
    "consonant_skeleton_sim",
    "jaro_winkler_addr",
    "levenshtein_ratio_addr",
    "token_set_ratio_addr",
    "addr_num_overlap_ratio",
    "addr_num_exact_match",
    "pincode_match_flag",
    "pincode_present",
    "blocking_score",
    "is_source2",
    "is_source3",
    "is_india",
    "is_us",
    "is_france",
    "name_token_jaccard",
    "addr_token_jaccard",
]

def _jaccard(s1: set, s2: set) -> float:
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)

def _extract_numbers(s: str) -> set:
    return set(re.findall(r"\b\d+\b", s))

def compute_pair_features(
    s1_name: str,
    s1_addr: str,
    cand_name: str,
    cand_addr: str,
    blocking_score: float,
    source: str,
    country: str,
) -> List[float]:
    """Computes 26 pairwise lexical, phonetic, and geographical features."""
    c_n1 = clean_name(s1_name)
    c_n2 = clean_name(cand_name)

    c_a1 = clean_address(s1_addr)
    c_a2 = clean_address(cand_addr)

    # 1-4. Name lexical similarities
    jw_name = jellyfish.jaro_winkler_similarity(c_n1, c_n2)
    lev_name = lev_ratio(c_n1, c_n2)

    toks1 = c_n1.split()
    toks2 = c_n2.split()
    sorted_n1 = " ".join(sorted(toks1))
    sorted_n2 = " ".join(sorted(toks2))
    tok_sort_name = lev_ratio(sorted_n1, sorted_n2)

    set1 = set(toks1)
    set2 = set(toks2)
    tok_set_name = _jaccard(set1, set2)

    # 5-8. Name structure
    len1 = len(c_n1)
    len2 = len(c_n2)
    name_len_diff = abs(len1 - len2)
    name_len_ratio = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 1.0
    exact_name = 1.0 if c_n1 == c_n2 and c_n1 else 0.0
    first_tok = 1.0 if toks1 and toks2 and toks1[0] == toks2[0] else 0.0

    # 9-11. Phonetics
    s1_sx = jellyfish.soundex(toks1[0]) if toks1 else ""
    c2_sx = jellyfish.soundex(toks2[0]) if toks2 else ""
    soundex_match = 1.0 if s1_sx and s1_sx == c2_sx else 0.0

    s1_meta = jellyfish.metaphone(toks1[0]) if toks1 else ""
    c2_meta = jellyfish.metaphone(toks2[0]) if toks2 else ""
    metaphone_match = 1.0 if s1_meta and s1_meta == c2_meta else 0.0

    sk1 = "".join([get_consonant_skeleton(t) for t in toks1])
    sk2 = "".join([get_consonant_skeleton(t) for t in toks2])
    sk_sim = lev_ratio(sk1, sk2) if sk1 and sk2 else 0.0

    # 12-14. Address similarities
    jw_addr = jellyfish.jaro_winkler_similarity(c_a1, c_a2)
    lev_addr = lev_ratio(c_a1, c_a2)
    tok_set_addr = _jaccard(set(c_a1.split()), set(c_a2.split()))

    # 15-16. Numeric identifiers in addresses (house/plot numbers)
    nums1 = _extract_numbers(c_a1)
    nums2 = _extract_numbers(c_a2)
    num_overlap = _jaccard(nums1, nums2)
    num_exact = 1.0 if nums1 and nums1 == nums2 else 0.0

    # 17-18. Pincode
    pin1 = extract_pincode(s1_addr)
    pin2 = extract_pincode(cand_addr)
    pin_present = 1.0 if pin1 and pin2 else 0.0
    pin_match = 1.0 if pin_present and pin1 == pin2 else (0.0 if pin_present else -1.0)

    # 19-24. Context and one-hot flags
    is_s2 = 1.0 if source == "source2" else 0.0
    is_s3 = 1.0 if source == "source3" else 0.0
    is_in = 1.0 if country.lower() == "india" else 0.0
    is_us = 1.0 if country.lower() == "us" else 0.0
    is_fr = 1.0 if country.lower() == "france" else 0.0

    # 25-26. Token Jaccard
    name_jaccard = _jaccard(set1, set2)
    addr_jaccard = _jaccard(set(c_a1.split()), set(c_a2.split()))

    return [
        jw_name,
        lev_name,
        tok_sort_name,
        tok_set_name,
        float(name_len_diff),
        name_len_ratio,
        exact_name,
        first_tok,
        soundex_match,
        metaphone_match,
        sk_sim,
        jw_addr,
        lev_addr,
        tok_set_addr,
        num_overlap,
        num_exact,
        pin_match,
        pin_present,
        float(blocking_score),
        is_s2,
        is_s3,
        is_in,
        is_us,
        is_fr,
        name_jaccard,
        addr_jaccard,
    ]

def extract_features_matrix(
    candidate_df: pl.DataFrame,
    s1_lookup: Dict[str, Tuple[str, str]],
    target_lookup: Dict[str, Tuple[str, str]],
) -> np.ndarray:
    """Computes 2D feature matrix (N x 26) from candidates DataFrame."""
    rows = candidate_df.select(["s1_id", "cand_id", "blocking_score", "source", "country"]).iter_rows(named=True)
    features = []

    for r in rows:
        sid = r["s1_id"]
        cid = r["cand_id"]
        s1_name, s1_addr = s1_lookup.get(sid, ("", ""))
        c_name, c_addr = target_lookup.get(cid, ("", ""))
        feats = compute_pair_features(
            s1_name,
            s1_addr,
            c_name,
            c_addr,
            r["blocking_score"],
            r["source"],
            r["country"],
        )
        features.append(feats)

    if not features:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
    return np.array(features, dtype=np.float32)
