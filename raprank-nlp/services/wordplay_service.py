import re
import logging
from typing import List, Tuple, Set, Dict
import nltk
from nltk.corpus import wordnet
import pronouncing

from services.language_utils import clean_word, content_lines, is_hindi_word, devanagari_to_roman, get_multilingual_stopwords
from config import scoring_config

logger = logging.getLogger(__name__)

# Ensure NLTK resources are downloaded
try:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    logger.warning("Failed to download NLTK resources: %s", e)

ALL_STOP_WORDS = get_multilingual_stopwords()

RAP_DOUBLE_ENTENDRE_WORDS: Set[str] = scoring_config.WORDPLAY["RAP_DOUBLE_ENTENDRE_WORDS"]

# Regex patterns for similes
# 1. "like [noun]" (checked programmatically for preceding pronouns, allowing optional articles)
# 2. "as [adj] as [noun]"
# 3. Hindi/Hinglish particles like jaise, jaisa, jaisi, tarah
SIMILE_LIKE_RE = re.compile(r"\blike\s+(?:a\s+|an\s+|the\s+|my\s+|your\s+|our\s+)?([a-zA-Z\u0900-\u097F]+)\b", re.IGNORECASE)
SIMILE_AS_AS_RE = re.compile(r"\bas\s+\w+\s+as\s+(\w+)\b", re.IGNORECASE)
SIMILE_HINDI_RE = re.compile(r"\b(\w+)\s+(jaise|jaisa|jaisi|tarah|जैसे|जैसा|जैसी|तरह)\b", re.IGNORECASE)
SIMILE_HINDI_PRE_RE = re.compile(r"\b(jaise|jaisa|jaisi|जैसे|जैसा|जैसी)\s+(?:[a-zA-Z\u0900-\u097F]+\s+){0,2}([a-zA-Z\u0900-\u097F]+)\b", re.IGNORECASE)

# Regex patterns for copula metaphors (e.g. "I am a beast", "main hoon aag")
METAPHOR_IM_RE = re.compile(r"\b(i\s+am|i\'m|you\s+are|you\'re|he\s+is|he\'s|she\s+is|she\'s|we\s+are|we\'re)\s+(?:a\s+|an\s+|the\s+)?([a-zA-Z\u0900-\u097F]+)", re.IGNORECASE)
METAPHOR_HINDI_RE = re.compile(r"\b(main\s+hoon|tu\s+hai|woh\s+hai|मैं\s+हूँ|तू\s+है|वह\s+है)\s+(?:ek\s+)?([a-zA-Z\u0900-\u097F]+)", re.IGNORECASE)


def is_noun_or_adj_in_wordnet(word: str) -> bool:
    """
    Check if the word is a valid content noun or adjective in the dictionary (WordNet).
    Filters out grammatical function words, verbs, and adverbs.
    """
    w_lower = word.lower()
    
    # Exclude standard stop words/grammatical particles
    if w_lower in ALL_STOP_WORDS:
        return False
        
    # Allow proper nouns or known capitalized artist/brand names
    if word[0].isupper() or w_lower in {"leviathan", "popeye", "reshammiya", "louboutin", "popeye k"}:
        return True
        
    # Devanagari Hindi words (not in WordNet) - check if they are Devanagari character blocks
    # Devanagari unicode range: U+0900 to U+097F
    if any(0x0900 <= ord(char) <= 0x097F for char in word):
        devanagari_stops = {"है", "था", "के", "से", "में", "को", "पर", "भी", "तो", "तौ", "और", "या", "यह", "ये", "वह", "वो", "कर", "ने", "का", "की", "जो", "ही", "ना", "न", "कौन", "क्या", "कब", "कहाँ", "क्यों", "कैसे"}
        return w_lower not in devanagari_stops
        
    synsets = wordnet.synsets(w_lower)
    if not synsets:
        # If it's a Romanized Hinglish word not in WordNet, allow if it's not a stop word
        return True
        
    # Check if the word has any noun ('n') or adjective ('a', 's') senses.
    # If it ONLY has verb ('v') or adverb ('r') senses, then it's NOT a valid metaphor/simile target.
    allowed_pos = {'n', 'a', 's'}
    has_allowed = any(syn.pos() in allowed_pos for syn in synsets)
    has_disallowed_only = all(syn.pos() in {'v', 'r'} for syn in synsets)
    
    return has_allowed and not has_disallowed_only


def detect_similes_and_metaphors(lyrics: str) -> Tuple[int, int, List[str], List[str]]:
    """
    Scan lyrics for similes and metaphors, validating comparison targets using WordNet.
    Returns (simile_count, metaphor_count, simile_matches, metaphor_matches).
    """
    similes = []
    metaphors = []
    
    for line in content_lines(lyrics):
        # Clean line of brackets/ad-libs
        clean_line = re.sub(r"\[.*?\]|\(.*?\)", "", line).strip()
        if not clean_line:
            continue
            
        # Similes: finditer to check preceding word programmatically and validate target
        for m in SIMILE_LIKE_RE.finditer(clean_line):
            start_idx = m.start()
            word = m.group(1)
            # Find the word preceding "like"
            preceding_part = clean_line[:start_idx].strip()
            preceding_words = preceding_part.split()
            preceding_word = preceding_words[-1].lower() if preceding_words else ""
            preceding_word = re.sub(r"[^\w]", "", preceding_word)
            
            if preceding_word not in ALL_STOP_WORDS and word.lower() not in ALL_STOP_WORDS:
                if is_noun_or_adj_in_wordnet(word):
                    similes.append(f"like {word}")
                
        for match in SIMILE_AS_AS_RE.findall(clean_line):
            if is_noun_or_adj_in_wordnet(match):
                similes.append(f"as... as {match}")
            
        for match in SIMILE_HINDI_RE.findall(clean_line):
            word, particle = match
            if word.lower() not in ALL_STOP_WORDS:
                if is_noun_or_adj_in_wordnet(word):
                    similes.append(f"{word} {particle}")

        for match in SIMILE_HINDI_PRE_RE.findall(clean_line):
            particle, word = match
            if word.lower() not in ALL_STOP_WORDS:
                if is_noun_or_adj_in_wordnet(word):
                    similes.append(f"{particle} {word}")
                
        # Metaphors: validate noun/adjective target
        for match in METAPHOR_IM_RE.findall(clean_line):
            pronoun_verb, word = match
            w_lower = word.lower()
            if w_lower not in ALL_STOP_WORDS and not w_lower.endswith("ing"):
                if is_noun_or_adj_in_wordnet(word):
                    metaphors.append(f"{pronoun_verb} {word}")
                
        for match in METAPHOR_HINDI_RE.findall(clean_line):
            prefix, word = match
            w_lower = word.lower()
            if w_lower not in ALL_STOP_WORDS:
                if is_noun_or_adj_in_wordnet(word):
                    metaphors.append(f"{prefix} {word}")
                
    return len(similes), len(metaphors), similes, metaphors


# Phonetically similar sounds for near-homophones (voicing / dental shifts)
SIMILAR_CONSONANTS = [
    {"T", "D", "DH", "TH"},
    {"P", "B", "F", "V"},
    {"K", "G"},
    {"S", "Z", "SH", "ZH"},
    {"M", "N", "NG"}
]


def edit_distance(l1: list, l2: list) -> int:
    m, n = len(l1), len(l2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if l1[i-1] == l2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]


def are_near_homophones(p1: str, p2: str) -> bool:
    """Check if two pronunciations are identical or very close (near-homophones)."""
    p1_clean = re.sub(r"\d+", "", p1).split()
    p2_clean = re.sub(r"\d+", "", p2).split()
    
    if not p1_clean or not p2_clean:
        return False
        
    if p1_clean == p2_clean:
        return True
        
    dist = edit_distance(p1_clean, p2_clean)
    if dist == 1:
        # If edit distance is 1, require starting phone to match to prevent simple rhymes
        if p1_clean[0] == p2_clean[0]:
            return True
        # Or if the difference is a single consonant within the same phonetic group
        if len(p1_clean) == len(p2_clean):
            diff_indices = [i for i, (a, b) in enumerate(zip(p1_clean, p2_clean)) if a != b]
            if len(diff_indices) == 1:
                phone_a = p1_clean[diff_indices[0]]
                phone_b = p2_clean[diff_indices[0]]
                for s in SIMILAR_CONSONANTS:
                    if phone_a in s and phone_b in s:
                        return True
                        
    return False


def hinglish_to_phones(word: str) -> str:
    """Generate basic phonetic keys for Hindi/Hinglish words to allow phonetic matching."""
    if is_hindi_word(word):
        word = devanagari_to_roman(word)
    word = word.lower()
    
    # Simplify vowels and consonant clusters
    word = re.sub(r"aa", "AH", word)
    word = re.sub(r"ee", "IY", word)
    word = re.sub(r"oo", "UW", word)
    word = re.sub(r"ai", "EH", word)
    
    word = re.sub(r"kh", "K", word)
    word = re.sub(r"gh", "G", word)
    word = re.sub(r"ch", "CH", word)
    word = re.sub(r"jh", "JH", word)
    word = re.sub(r"th", "T", word)
    word = re.sub(r"dh", "D", word)
    word = re.sub(r"ph", "F", word)
    word = re.sub(r"bh", "B", word)
    word = re.sub(r"sh", "SH", word)
    
    mapping = {
        'a': 'AH', 'e': 'EH', 'i': 'IH', 'o': 'OW', 'u': 'AH',
        'k': 'K', 'g': 'G', 'c': 'K', 'j': 'JH', 't': 'T', 'd': 'D',
        'p': 'P', 'b': 'B', 'f': 'F', 'v': 'V', 's': 'S', 'z': 'Z',
        'r': 'R', 'l': 'L', 'm': 'M', 'n': 'N', 'h': 'HH', 'y': 'Y', 'w': 'W'
    }
    
    phones = []
    i = 0
    while i < len(word):
        if word[i].isupper():
            val = ""
            while i < len(word) and word[i].isupper():
                val += word[i]
                i += 1
            phones.append(val)
        else:
            ch = word[i]
            if ch in mapping:
                phones.append(mapping[ch])
            i += 1
            
    return " ".join(phones)


def verify_pun_meanings(w1: str, w2: str) -> bool:
    """Verify using WordNet that the two homophonic words have different semantic meanings."""
    try:
        s1 = wordnet.synsets(w1.lower())
        s2 = wordnet.synsets(w2.lower())
        if s1 and s2:
            return s1[0] != s2[0]
        return True
    except Exception:
        return True


def is_stop_word(word: str) -> bool:
    """Helper to check if a word is a stop word/grammatical particle."""
    return word.lower() in ALL_STOP_WORDS


def detect_homophone_puns(lyrics: str) -> Tuple[int, List[str]]:
    """
    Find homophones and near-homophones in close proximity to identify potential puns.
    Returns (pun_count, list of pun word pairs).
    """
    lines = content_lines(lyrics)
    puns = []
    
    # Pre-calculate phones for all words in lyrics
    word_phones: Dict[str, str] = {}
    for line in lines:
        for w in line.split():
            cw = clean_word(w)
            if cw and len(cw) > 1:
                cw_lower = cw.lower()
                if cw_lower not in word_phones:
                    phones_list = pronouncing.phones_for_word(cw_lower)
                    if phones_list:
                        word_phones[cw_lower] = phones_list[0]
                    else:
                        word_phones[cw_lower] = hinglish_to_phones(cw)

    # Scan lines in a rolling window of 3 lines
    for idx in range(len(lines)):
        window_lines = lines[idx:idx + 3]
        window_text = " ".join(window_lines).lower()
        window_words = [clean_word(w) for w in window_text.split() if clean_word(w)]
        window_words = list(set(w for w in window_words if w and len(w) > 1))
        
        for i in range(len(window_words)):
            for j in range(i + 1, len(window_words)):
                w1 = window_words[i]
                w2 = window_words[j]
                if w1 == w2:
                    continue
                # Ensure at least one word in the pairing is a content word (not a stop word/grammatical particle)
                if is_stop_word(w1) and is_stop_word(w2):
                    continue
                p1 = word_phones.get(w1)
                p2 = word_phones.get(w2)
                if p1 and p2:
                    if are_near_homophones(p1, p2):
                        if verify_pun_meanings(w1, w2):
                            pair_str = " vs ".join(sorted([w1, w2]))
                            if pair_str not in puns:
                                puns.append(pair_str)
                            
    return len(puns), puns


def detect_double_entendres(lyrics: str) -> Tuple[int, List[str]]:
    """
    Use NLTK WordNet to find highly polysemous words (words with multiple distinct meanings/senses)
    that are used in the lyrics, cross-referencing with rap-specific vocabulary.
    Returns (double_entendre_count, list of polysemous words).
    """
    entendres = []
    
    for line in content_lines(lyrics):
        for raw in line.split():
            w = clean_word(raw)
            if not w or len(w) <= 2 or w in ALL_STOP_WORDS:
                continue
                
            w_lower = w.lower()
            
            try:
                # Query WordNet for synsets (senses)
                synsets = wordnet.synsets(w_lower)
                num_senses = len(synsets)
                
                # A word is counted as a double-entendre candidate if:
                # 1. It is a known rap polysemous word (e.g. bar, key, plate) AND has >= senses from config.
                # 2. Or it is an extremely polysemous word in standard English (>= senses from config).
                min_rap = scoring_config.WORDPLAY["ENTENDRE_MIN_SENSES_RAP"]
                min_gen = scoring_config.WORDPLAY["ENTENDRE_MIN_SENSES_GENERAL"]
                if (w_lower in RAP_DOUBLE_ENTENDRE_WORDS and num_senses >= min_rap) or num_senses >= min_gen:
                    if w_lower not in entendres:
                        entendres.append(w_lower)
            except Exception:
                # Fallback to simple set check if WordNet fails or is unavailable
                if w_lower in RAP_DOUBLE_ENTENDRE_WORDS:
                    if w_lower not in entendres:
                        entendres.append(w_lower)
                        
    return len(entendres), entendres


def calculate(lyrics: str) -> Tuple[float, Dict]:
    """
    Calculate the overall Wordplay Score (0-100) and return breakdown details.
    """
    simile_count, metaphor_count, similes, metaphors = detect_similes_and_metaphors(lyrics)
    pun_count, puns = detect_homophone_puns(lyrics)
    double_entendre_count, double_entendres = detect_double_entendres(lyrics)
    
    lines = content_lines(lyrics)
    num_lines = max(len(lines), 1)
    
    # Calculate density scores
    simile_density = simile_count / num_lines
    metaphor_density = metaphor_count / num_lines
    pun_density = pun_count / num_lines
    entendre_density = double_entendre_count / num_lines
    
    # Compute sub-scores (0-100 scale) dynamically from config
    targets = scoring_config.WORDPLAY["ELITE_TARGETS"]
    simile_score = min((simile_density / targets["simile"]) * 100.0, 100.0)
    metaphor_score = min((metaphor_density / targets["metaphor"]) * 100.0, 100.0)
    pun_score = min((pun_density / targets["pun"]) * 100.0, 100.0)
    entendre_score = min((entendre_density / targets["entendre"]) * 100.0, 100.0)
    
    # Dynamic wordplay logic:
    # Exceling in a single category should not be penalized by a lack of others.
    # Therefore, the score is calculated as weighted combination of overall density
    # and the maximum sub-score achieved in any single category.
    total_elements = simile_count + metaphor_count + pun_count + double_entendre_count
    total_density = total_elements / num_lines
    
    overall_density_score = scoring_config.evaluate_piecewise_curve(
        total_density,
        scoring_config.WORDPLAY["CURVE_THRESHOLDS"],
        scoring_config.WORDPLAY["CURVE_SCORES"]
    )
        
    max_sub_score = max(simile_score, metaphor_score, pun_score, entendre_score)
    
    w_overall = scoring_config.WORDPLAY["WEIGHT_OVERALL_DENSITY"]
    w_max = scoring_config.WORDPLAY["WEIGHT_MAX_SUB_SCORE"]
    wordplay_score = overall_density_score * w_overall + max_sub_score * w_max
    
    metadata = {
        "wordplay_score": round(wordplay_score, 2),
        "simile_count": simile_count,
        "metaphor_count": metaphor_count,
        "puns_count": pun_count,
        "double_entendres_count": double_entendre_count,
        "similes": similes,
        "metaphors": metaphors,
        "puns": puns,
        "double_entendres": double_entendres
    }
    
    return round(wordplay_score, 2), metadata
