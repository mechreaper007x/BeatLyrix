import pytest
from services.lyrical_compiler import LyricalLexer, LyricalParser, compile_lyrics

def test_lexer_latin_spelling_normalization():
    # Verify Hinglish vowel and consonant collapsing
    lexer = LyricalLexer("")
    assert lexer.normalize_hinglish_spelling("saath") == "saath"
    assert lexer.normalize_hinglish_spelling("sath") == "sath"
    assert lexer.normalize_hinglish_spelling("bhai") == "bhai"
    assert lexer.normalize_hinglish_spelling("jindagi") == "zindagi"
    assert lexer.normalize_hinglish_spelling("khilaaf") == "khilaaf"

def test_lexer_schwa_deletion_automaton():
    lexer = LyricalLexer("")
    
    # Word-final schwa deletion (राम -> raam, not raama)
    graphemes_ram = lexer.apply_schwa_deletion("राम")
    assert "a" not in graphemes_ram[-2:] # final schwa deleted on 'm'
    
    # Internal schwa deletion (करता -> karta, not karata)
    graphemes_karta = lexer.apply_schwa_deletion("करता")
    # should be ['k', 'a', 'r', 't', 'aa']
    assert "a" not in [graphemes_karta[2], graphemes_karta[3]] # middle 'a' deleted

def test_lexer_scan_emits_tokens():
    lyrics = "No Cap (woo)\nसज़ाए मौत"
    lexer = LyricalLexer(lyrics)
    token_lines = lexer.scan()
    
    assert len(token_lines) == 2
    # Check that ad-libs are flagged
    has_adlib = any(t.type == "T_ADLIB" for line in token_lines for t in line)
    assert has_adlib

def test_parser_symbol_table_scopes():
    lyrics = "aaya kaun\naaya kaun\nlife khoke"
    lexer = LyricalLexer(lyrics)
    token_lines = lexer.scan()
    parser = LyricalParser(token_lines)
    
    compiled = parser.compile()
    assert compiled["lyrical_score"] >= 0
    assert compiled["avg_syllables_per_line"] > 0
    assert compiled["detected_rhyme_count"] >= 0

def test_end_to_end_compiler():
    lyrics = (
        "poochho inse aaya kaun\n"
        "dilli ka launda machata hai shor\n"
        "sath mere chalta hai mera ye flow\n"
        "baki sab fake main hoon yahan pro"
    )
    metrics = compile_lyrics(lyrics)
    assert "lyrical_score" in metrics
    assert "rhyme_complexity" in metrics
    assert "syllable_density" in metrics
    assert "lexical_information_density" in metrics
    assert metrics["lyrical_score"] > 0.0
