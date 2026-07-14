"""
Lyrical Lexer & Parser Compiler (LLPC) Framework v1
Treats rap lyrics as source code, compiling them into phonetic tokens,
analyzing syllable scopes, and parsing rhyme state transitions via Markov chains.
"""
from __future__ import annotations

import re
import math
import logging
from typing import TypedDict, List, Dict, Tuple, Set

from config import scoring_config

logger = logging.getLogger(__name__)

# ── Unified Phonetic Constants ──────────────────────────────────────────────
VOWEL_CLASSES = {
    "A": ["a", "aa", "ah", "ae", "ay", "ा"],
    "I": ["i", "ee", "y", "ey", "ि", "ी"],
    "U": ["u", "oo", "ou", "ु", "ू"],
    "E": ["e", "ai", "े", "ै"],
    "O": ["o", "au", "ो", "ौ"]
}

CONSONANT_CLASSES = {
    "G1": ["k", "kh", "g", "gh", "q", "क", "ख", "ग", "घ", "क़", "ख़", "ग़"],  # Velars
    "G2": ["ch", "chh", "j", "jh", "z", "च", "छ", "ज", "झ", "ज़"],           # Palatals / Fricatives
    "G3": ["t", "th", "d", "dh", "n", "ट", "ठ", "ड", "ढ", "ण", "त", "थ", "द", "ध", "न", "ड़", "ढ़"], # Dentals / Alveolars
    "G4": ["p", "ph", "b", "bh", "m", "f", "प", "फ", "ब", "भ", "म", "फ़"],   # Labials
    "G5": ["y", "r", "l", "v", "w", "sh", "s", "h", "य", "र", "ल", "व", "श", "ष", "स", "ह"] # Semivowels / Liquids / Sibilants
}

# Invert lookup tables for O(1) matching during lexing
VOWEL_LOOKUP = {char: cls for cls, chars in VOWEL_CLASSES.items() for char in chars}
CONSONANT_LOOKUP = {char: cls for cls, chars in CONSONANT_CLASSES.items() for char in chars}

# Hindi Devanagari character boundaries
DEVA_START = 0x0900
DEVA_END = 0x097F

# ── Compiler Token Definitions ──────────────────────────────────────────────
class Token:
    def __init__(self, type_: str, value: str, char_pos: int = 0):
        self.type = type_      # 'T_VOWEL', 'T_CONSONANT', 'T_GRAMMAR_SUFFIX', 'T_ADLIB', 'T_NEWLINE'
        self.value = value      # Unified class key (e.g. 'A', 'G3') or actual text
        self.char_pos = char_pos

    def __repr__(self) -> str:
        return f"Token({self.type}, {self.value})"

# ── The Phonetic Lexer (DFA Scanner) ─────────────────────────────────────────
class LyricalLexer:
    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.length = len(text)

    def is_devanagari(self, char: str) -> bool:
        return DEVA_START <= ord(char) <= DEVA_END if char else False

    def normalize_hinglish_spelling(self, word: str) -> str:
        """Applies spelling normalization to Romanized Hinglish strings."""
        word = word.lower()
        # Normalization transformations
        word = re.sub(r"aa|ah|ae", "aa", word)
        word = re.sub(r"ee|ie|y", "ee", word)
        word = re.sub(r"oo|uu", "oo", word)
        word = re.sub(r"kh|q", "kh", word)
        word = re.sub(r"z|j", "z", word)
        return word

    def apply_schwa_deletion(self, devanagari_word: str) -> list[str]:
        """
        Linguistic DFA modeling Hindi final and internal Schwa Deletion.
        Returns a list of Romanized phonetic graphemes.
        """
        from services.language_utils import devanagari_to_roman
        roman = devanagari_to_roman(devanagari_word)
        
        # Parse Romanized string into phonetic clusters (consonants vs vowels)
        # Apply 3-syllable internal schwa deletion pattern
        # e.g., 'karata' -> 'karta', 'namakeen' -> 'namkeen'
        # Rule: C1 + 'a' + C2 + 'a' + C3 + full_vowel -> delete second 'a'
        pattern = r"([b-df-hj-np-tv-z]+)a([b-df-hj-np-tv-z]+)a([b-df-hj-np-tv-z]+)([aeiou]{2,})"
        roman = re.sub(pattern, r"\1a\2\3\4", roman)
        
        # Word-final schwa deletion: final 'a' is dropped if preceded by consonant
        if len(roman) > 1 and roman.endswith("a") and not roman.endswith("aa"):
            roman = roman[:-1]
            
        # Segment into grapheme units
        graphemes = []
        i = 0
        while i < len(roman):
            if i + 1 < len(roman) and roman[i:i+2] in ["kh", "gh", "ch", "jh", "th", "dh", "ph", "bh", "sh", "aa", "ee", "oo", "ai", "au"]:
                graphemes.append(roman[i:i+2])
                i += 2
            else:
                graphemes.append(roman[i])
                i += 1
        return graphemes

    def lex_word(self, word: str) -> list[Token]:
        """Convert a single cleaned word into a stream of phonological tokens."""
        tokens = []
        
        # Check if word contains Devanagari characters
        is_deva = any(self.is_devanagari(c) for c in word)
        
        if is_deva:
            # Run abugida G2P mapping with Schwa Deletion
            graphemes = self.apply_schwa_deletion(word)
        else:
            # Romanized / English word
            normalized = self.normalize_hinglish_spelling(word)
            graphemes = []
            i = 0
            while i < len(normalized):
                if i + 1 < len(normalized) and normalized[i:i+2] in ["kh", "gh", "ch", "jh", "th", "dh", "ph", "bh", "sh", "aa", "ee", "oo", "ai", "au"]:
                    graphemes.append(normalized[i:i+2])
                    i += 2
                else:
                    graphemes.append(normalized[i])
                    i += 1

        # Map graphemes to tokens
        for g in graphemes:
            if g in VOWEL_LOOKUP:
                tokens.append(Token("T_VOWEL", VOWEL_LOOKUP[g]))
            elif g in CONSONANT_LOOKUP:
                tokens.append(Token("T_CONSONANT", CONSONANT_LOOKUP[g]))
            else:
                # Fallback map characters individually
                for char in g:
                    if char in VOWEL_LOOKUP:
                        tokens.append(Token("T_VOWEL", VOWEL_LOOKUP[char]))
                    elif char in CONSONANT_LOOKUP:
                        tokens.append(Token("T_CONSONANT", CONSONANT_LOOKUP[char]))
        return tokens

    def scan(self) -> list[list[Token]]:
        """Scans the entire document line-by-line, returning token streams."""
        lines_tokens = []
        lines = self.text.split("\n")
        
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped or (line_stripped.startswith("[") and line_stripped.endswith("]")):
                continue
                
            line_tokens = []
            
            # Tokenize word-by-word
            words = line_stripped.split()
            for word in words:
                cleaned = re.sub(r"[^\w\u0900-\u097F]", "", word).strip()
                if not cleaned:
                    continue
                
                # Check ad-lib tags
                if (word.startswith("(") or word.endswith(")")):
                    line_tokens.append(Token("T_ADLIB", cleaned))
                    continue
                    
                word_tokens = self.lex_word(cleaned)
                line_tokens.extend(word_tokens)
                
            line_tokens.append(Token("T_NEWLINE", "\n"))
            lines_tokens.append(line_tokens)
            
        return lines_tokens

# ── The Symbol Table (Rhyme Sound Registry) ──────────────────────────────────
class RhymeSymbol:
    def __init__(self, key: str, syllable_span: int, start_line: int):
        self.key = key
        self.syllable_span = syllable_span
        self.occurrences: list[int] = [start_line]
        self.last_seen = start_line

class SymbolTable:
    def __init__(self):
        self.symbols: dict[str, RhymeSymbol] = {}

    def lookup_or_register(self, key: str, syllable_span: int, current_line: int) -> bool:
        """
        Returns True if a symbol is matched within active scope (scope limit of 3 lines).
        Otherwise, registers or updates the symbol and returns False.

        `current_line` must be a position in the RHYME STREAM (i.e. only
        lines that produced a rhyme key), not the raw lyric line index --
        section-header/blank lines (e.g. a bracketed "[छ]" divider) don't
        rhyme with anything and shouldn't burn scope width against a
        genuine nearby rhyme.
        """
        if key in self.symbols:
            symbol = self.symbols[key]
            if current_line - symbol.last_seen <= 3:
                symbol.occurrences.append(current_line)
                symbol.last_seen = current_line
                return True
            else:
                # Re-activate scope
                symbol.last_seen = current_line
                symbol.occurrences.append(current_line)
                return False
        else:
            self.symbols[key] = RhymeSymbol(key, syllable_span, current_line)
            return False

# ── The Lyrical Parser & Metric Compiler ──────────────────────────────────────
class LyricalParser:
    def __init__(self, token_lines: list[list[Token]]):
        self.token_lines = token_lines
        self.symbol_table = SymbolTable()

    def parse_syllables_and_rhyme_key(self, tokens: list[Token]) -> tuple[int, str]:
        """
        Parser rule: Reduces a token stream of a line to its syllable count
        and extracts the final phonetic rhyme signature (nuclei + codas).

        Cross-script normalisation: Devanagari vowel matras map to the same
        abstract vowel class as their Latin counterparts, so a Hindi line
        ending in /kaun/ and an English line ending in /on/ share rhyme key
        O-G1 and are correctly counted as a rhyme match.
        """
        # Cross-script vowel unification map (Devanagari matra -> Latin class)
        # VOWEL_CLASSES already unifies these into the same keys (A/I/U/E/O),
        # so the Token values are already normalised -- no extra step needed
        # at this level. The fix is ensuring Devanagari words go through
        # apply_schwa_deletion -> VOWEL_LOOKUP, which maps ा->A, ि->I, etc.
        # That path is already correct in lex_word. What was missing was that
        # English words with vowel combos like 'au', 'ou' -> class O, and
        # Devanagari ौ -> class O -- so they already share the key. The
        # remaining gap is short-vowel English words where normalization
        # collapses 'a','aa','ah' all to class A, matching ा which is also A.
        vowel_tokens = [t for t in tokens if t.type == "T_VOWEL"]
        syllables = len(vowel_tokens)

        # Build rhyme signature from the final vowel class + final trailing
        # consonant (a true end-rhyme signature, matching how the ear judges
        # a rhyme). Requiring the last TWO vowels and TWO consonants to match
        # (the previous scheme) fragments genuine end-rhymes into distinct
        # keys whenever the second-to-last syllable differs -- this
        # systematically undercounts songs that rhyme through a wide set of
        # end sounds (single-syllable-anchored schemes) relative to songs
        # that repeat the exact same two-syllable tail, which is a stylistic
        # difference, not a rhyme-skill difference.
        if syllables >= 1:
            last_vowel = vowel_tokens[-1].value
            trailing_consonants = [t.value for t in tokens if t.type == "T_CONSONANT"][-1:]
            rhyme_key = last_vowel + "|" + "-".join(trailing_consonants)
        else:
            rhyme_key = "EMPTY"

        return max(1, syllables), rhyme_key

    def calculate_rhyme_transition_entropy(self, rhyme_stream: list[str]) -> float:
        """
        Computes the Markov Chain Transition Entropy over the sequence of rhyme sounds.
        Highly cohesive, lyrical structures have low transition entropy.
        """
        if len(rhyme_stream) <= 1:
            return 1.0  # Maximum entropy (no structural data)
            
        # Get unique states
        states = list(set(rhyme_stream))
        state_idx = {state: idx for idx, state in enumerate(states)}
        num_states = len(states)
        
        # Construct transition count matrix
        transitions = [[0.0] * num_states for _ in range(num_states)]
        state_counts = [0.0] * num_states
        
        for i in range(len(rhyme_stream) - 1):
            curr_state = rhyme_stream[i]
            next_state = rhyme_stream[i+1]
            c_idx = state_idx[curr_state]
            n_idx = state_idx[next_state]
            transitions[c_idx][n_idx] += 1.0
            state_counts[c_idx] += 1.0

        # Compute Transition Probabilities
        entropy = 0.0
        stationary_distribution = [count / sum(state_counts) if sum(state_counts) > 0 else 0.0 for count in state_counts]
        
        for i in range(num_states):
            row_sum = state_counts[i]
            if row_sum == 0:
                continue
            row_entropy = 0.0
            for j in range(num_states):
                prob = transitions[i][j] / row_sum
                if prob > 0:
                    row_entropy -= prob * math.log2(prob)
            entropy += stationary_distribution[i] * row_entropy
            
        return entropy

    def compile(self, lyrics: str = "") -> dict[str, float]:
        """Compiles the parsed lyrics stream, returning the unsupervised score metrics."""
        rhyme_stream = []
        syllable_counts = []
        adlib_count = 0
        match_count = 0
        total_lines = len(self.token_lines)
        
        if total_lines == 0:
            return {
                "lyrical_score": 0.0,
                "rhyme_complexity": 0.0,
                "syllable_density": 0.0,
                "lexical_information_density": 0.0
            }

        # Scan line-by-line
        for line_idx, tokens in enumerate(self.token_lines):
            # Track ad-libs
            adlib_count += sum(1 for t in tokens if t.type == "T_ADLIB")
            
            # Parse syllable count and rhyme key
            syl_count, rhyme_key = self.parse_syllables_and_rhyme_key(tokens)
            syllable_counts.append(syl_count)
            
            if rhyme_key != "EMPTY":
                rhyme_stream.append(rhyme_key)

                # Check Symbol Table -- use the rhyme-stream-relative position
                # (not the raw lyric line_idx) so blank/section-header lines
                # don't silently shrink the 3-line active scope.
                is_match = self.symbol_table.lookup_or_register(rhyme_key, 2, len(rhyme_stream) - 1)
                if is_match:
                    match_count += 1

        # 1. Rhyme Transition Entropy (Markov Chains)
        entropy = self.calculate_rhyme_transition_entropy(rhyme_stream)
        unique_states = len(set(rhyme_stream))
        max_entropy = math.log2(max(2, unique_states))
        normalized_entropy = min(1.0, entropy / max_entropy) if max_entropy > 0 else 0.0
        
        # Rhyme density = matches per rhyming line
        rhyme_density = match_count / max(1, len(rhyme_stream))
        
        # A complex rapper has both high rhyme density (frequent rhyming) and high entropy (transitioning between varied rhyme sounds/states).
        # Yalgaar has high density but very low entropy.
        # A bad rapper might have low density and low entropy.
        # A great rapper has high density and high entropy.
        # rhyme_density and normalized_entropy are raw fractions that never
        # approach 1.0 across real lyrics (ECDF on the 302-track corpus put
        # their 99th percentiles at ~0.57 and ~0.32) -- blending them as raw
        # fractions compressed rhyme_complexity to a max of ~43/100 even for
        # the best real track. Map each through its own percentile-fit curve
        # first (config.LLPC_RHYME), then blend with the same 0.40/0.60 ratio.
        _llpc_cfg = scoring_config.LLPC_RHYME
        density_score = scoring_config.evaluate_piecewise_curve(
            rhyme_density, _llpc_cfg["DENSITY_THRESHOLDS"], _llpc_cfg["DENSITY_SCORES"]
        )
        entropy_score = scoring_config.evaluate_piecewise_curve(
            normalized_entropy, _llpc_cfg["ENTROPY_THRESHOLDS"], _llpc_cfg["ENTROPY_SCORES"]
        )
        rhyme_complexity = density_score * 0.40 + entropy_score * 0.60
        
        # 2. Syllable Density & Variance
        avg_syllables = sum(syllable_counts) / len(syllable_counts)
        if avg_syllables < 6.0:
            line_length_score = (avg_syllables / 6.0) * 40.0
        elif avg_syllables <= 14.0:
            line_length_score = 40.0 + ((avg_syllables - 6.0) / 8.0) * 55.0
        else:
            line_length_score = max(50.0, 95.0 - (avg_syllables - 14.0) * 5.0)

        # Count syllables per word and polysyllabic word percentage
        all_words_syllables = []
        polysyl_words = 0
        if lyrics:
            lines = lyrics.split("\n")
            for line in lines:
                line_stripped = line.strip()
                if not line_stripped or (line_stripped.startswith("[") and line_stripped.endswith("]")):
                    continue
                words = line_stripped.split()
                for word in words:
                    cleaned = re.sub(r"[^\w\u0900-\u097F]", "", word).strip()
                    if not cleaned:
                        continue
                    # Lex single word
                    word_lexer = LyricalLexer(cleaned)
                    word_tokens = word_lexer.lex_word(cleaned)
                    syls = len([t for t in word_tokens if t.type == "T_VOWEL"])
                    syls = max(1, syls)
                    all_words_syllables.append(syls)
                    if syls >= 3:
                        polysyl_words += 1

        avg_syls_per_word = sum(all_words_syllables) / len(all_words_syllables) if all_words_syllables else 1.2
        polysyl_ratio = polysyl_words / max(1, len(all_words_syllables))

        # Map average syllables per word (SPW)
        word_density_score = min(100.0, max(0.0, (avg_syls_per_word - 1.1) / 0.6) * 100.0)
        # Map polysyllabic ratio
        polysyl_score = min(100.0, (polysyl_ratio / 0.15) * 100.0)

        # Combine: 60% SPW, 20% polysyl ratio, 20% line length score
        syllable_score = (word_density_score * 0.60 + polysyl_score * 0.20 + line_length_score * 0.20)

        # 3. Lexical Information Density
        total_tokens = sum(len(line) for line in self.token_lines)
        distinct_phoneme_tokens = len(set(t.value for line in self.token_lines for t in line if t.type in ["T_VOWEL", "T_CONSONANT"]))
        lex_density = (distinct_phoneme_tokens / max(1, total_tokens)) * 1000.0
        lexical_score = min(100.0, (lex_density / 120.0) * 100.0)

        # 4. Composite Lyrical Quality Index (LQI)
        # Base formula: 50% rhyme_complexity + 25% syllable_density + 25% lexical_density
        lqi = (rhyme_complexity * 0.50) + (syllable_score * 0.25) + (lexical_score * 0.25)

        # 5. Vocabulary Richness Bonus (thresholds from config.VOCAB_BONUS)
        # Artists with near-zero word repetition are penalised by the rhyme density
        # component because flow-pocketing deliberately avoids repeated end-words.
        # We derive a proxy MSTTR from the token stream (distinct content-word ratio).
        from config.scoring_config import VOCAB_BONUS as _VB
        _seg  = VOCABULARY.get("MSTTR_SEGMENT_SIZE", 50) if False else 50  # imported above
        _threshold = _VB["MSTTR_THRESHOLD"]
        _max_bonus = _VB["MAX_BONUS_POINTS"]
        _min_words = _VB["MIN_WORDS_REQUIRED"]
        _step_frac = _VB["SEGMENT_STEP_FRACTION"]

        if lyrics:
            content_words = [
                w.lower() for line in lyrics.split("\n")
                for w in line.split()
                if not (w.startswith("[") or w.endswith("]"))
                and len(re.sub(r"[^\w]", "", w)) > 1
            ]
            if len(content_words) >= _min_words:
                segment_size = 50
                step = max(1, int(segment_size * _step_frac))
                ttr_scores = []
                for start in range(0, len(content_words) - segment_size + 1, step):
                    segment = content_words[start:start + segment_size]
                    ttr_scores.append(len(set(segment)) / len(segment))
                proxy_msttr = sum(ttr_scores) / len(ttr_scores) if ttr_scores else 0.0

                if proxy_msttr > _threshold:
                    vocab_bonus = ((proxy_msttr - _threshold) / (1.0 - _threshold)) * _max_bonus
                    lqi = min(100.0, lqi + vocab_bonus)

        return {
            "lyrical_score": round(lqi, 2),
            "rhyme_complexity": round(rhyme_complexity, 2),
            "syllable_density": round(syllable_score, 2),
            "lexical_information_density": round(lexical_score, 2),
            "avg_syllables_per_line": round(avg_syllables, 2),
            "detected_rhyme_count": match_count,
            "unique_rhyme_schemes": unique_states,
            "rhyme_entropy": round(entropy, 4),
            # Raw pre-blend components of rhyme_complexity (see line ~335),
            # exposed for corpus/calibrate.py ECDF fitting of the 0.40/0.60
            # blend weights -- percentiles of rhyme_complexity itself would
            # just re-derive the blend's own shape, not validate its weights.
            "rhyme_density": round(rhyme_density, 4),
            "normalized_entropy": round(normalized_entropy, 4),
        }

# ── Integration Interface ───────────────────────────────────────────────────
def compile_lyrics(lyrics: str) -> dict[str, float]:
    """Main compilation entry point for the Lyrical Lexer & Parser Compiler (LLPC)."""
    lexer = LyricalLexer(lyrics)
    token_lines = lexer.scan()
    parser = LyricalParser(token_lines)
    return parser.compile(lyrics)
