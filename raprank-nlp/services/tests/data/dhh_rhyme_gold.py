# -*- coding: utf-8 -*-
"""
DHH rhyme gold set -- the oracle for the from-scratch Hindi/Hinglish phoneme
engine (services/dhh_phonemes.py, see docs/dhh_dictionary/PLAN.md).

Every entry is a fact about PRONUNCIATION, not spelling. The current
spelling-heuristic keyer (rhyme_service.normalize_hinglish and the Devanagari
last-N-chars key) gets many of these wrong -- most importantly anything
involving schwa deletion (घर is "ghar", never "ghara") and vowel length
(naam vs nam do NOT rhyme). The phoneme engine must pass ALL of these before
it can replace the heuristic.

Sections:
  GOLD_POSITIVE       -- word pairs whose FINAL syllable rhymes (same
                         rhyme_key: last vowel nucleus + trailing consonants).
  GOLD_NEGATIVE       -- near-spellings that do NOT rhyme (different final
                         nucleus and/or coda). Vowel-length traps live here.
  GOLD_MULTI_POSITIVE -- pairs sharing the last TWO vowel nuclei onward
                         (same multi_rhyme_key), the multisyllabic axis.
  GOLD_MULTI_NEGATIVE -- pairs whose final syllable may match but whose
                         2-nucleus key must NOT.
  GOLD_SCRIPT_EQUIV   -- (devanagari, [romanized variants...]): every variant
                         must produce the SAME phoneme sequence as the
                         Devanagari form. This is what collapses hoon/hun.

Romanized forms are the spellings that actually occur in the rap corpus
(Latin-script Hinglish), not scholarly IAST.
"""

# ── Final-syllable rhymes (schwa deletion makes most of these work) ──────────
# Format: (word_a, word_b) -- rhyme_key(a) == rhyme_key(b), both scripts OK.
GOLD_POSITIVE = [
    # final schwa deletion -> consonant-final rhymes (the core failure of the
    # old keyer: spelled forms end in inherent 'a' that speech drops)
    ("घर", "डर"),            # ghar / dar        -> -ar
    ("ghar", "par"),
    ("दिल", "मिल"),          # dil / mil         -> -il
    ("dil", "mushkil"),
    ("कल", "चल"),            # kal / chal        -> -al
    ("दर्द", "मर्द"),          # dard / mard       -> -ard
    ("नज़र", "मगर"),          # nazar / magar     -> -ar
    ("safar", "nazar"),
    ("asar", "safar"),
    ("लहर", "शहर"),          # lahar / shahar    -> -ahar
    ("समझ", "समझ"),          # identity sanity: schwa handling is stable
    ("fikr", "zikr"),         # -ikr (common Urdu pair in rap)
    # long-vowel + coda
    ("रात", "बात"),          # raat / baat       -> -aat
    ("baat", "saat"),
    ("हाथ", "साथ"),          # haath / saath     -> -aath
    ("यार", "प्यार"),         # yaar / pyaar      -> -aar
    ("pyaar", "izhaar"),
    ("शाम", "नाम"),          # shaam / naam      -> -aam
    ("naam", "kaam"),
    ("जान", "शान"),          # jaan / shaan      -> -aan
    ("ख्वाब", "किताब"),       # khwaab / kitaab   -> -aab
    ("jawaab", "hisaab"),
    ("khwaab", "kharaab"),
    ("झूठ", "लूट"),           # jhooth / loot     -> uu + retroflex-ish coda
    ("आग", "राग"),           # aag / raag        -> -aag
    ("log", "rog"),
    ("राह", "चाह"),          # raah / chaah      -> -aah
    # nasalized
    ("जहाँ", "वहाँ"),         # jahaan / wahaan   -> nasal -aa~
    ("jahaan", "yahaan"),
    # vowel-final (final vowel is real, not schwa)
    ("गली", "चली"),          # gali / chali      -> -ali/-alii
    ("पानी", "कहानी"),        # paani / kahaani   -> -nii
    ("सोना", "खोना"),         # sona / khona      -> -onaa
    ("rona", "sona"),
    ("जीना", "सीना"),         # jeena / seena     -> -iinaa
    ("पैसा", "कैसा"),         # paisa / kaisa     -> -aisaa
    ("भारी", "सारी"),         # bhaari / saari    -> -aarii
    ("yaari", "bimaari"),
    ("गाना", "ज़माना"),        # gaana / zamaana   -> -aanaa
    ("रोटी", "छोटी"),         # roti / chhoti     -> -oTii
    # -at cluster (Urdu-origin, ubiquitous in DHH)
    ("मेहनत", "इज़्ज़त"),        # mehnat / izzat    -> -at
    ("aadat", "mohabbat"),
    ("kismat", "mannat"),
    # medial schwa deletion feeding the rhyme (apanaa->apnaa)
    ("सपना", "अपना"),         # sapna / apna      -> -apnaa
    ("मरना", "करना"),         # marna / karna     -> -arnaa
    ("girna", "phirna"),
]

# ── NOT rhymes (the engine must keep these apart) ────────────────────────────
# Vowel length, different nuclei, and suffix traps.
GOLD_NEGATIVE = [
    ("दिल", "दाल"),          # dil / daal    -- i vs aa
    ("dil", "dal"),           # i vs a
    ("कल", "काल"),           # kal / kaal    -- a vs aa (length matters)
    ("chal", "chaal"),
    ("नाम", "नम"),           # naam / nam    -- aa vs a
    ("रात", "रीत"),          # raat / reet   -- aa vs ii
    ("baat", "bhoot"),        # aa vs uu
    ("सोना", "सेना"),         # sona / sena   -- o vs e
    ("jeena", "jaana"),       # ii vs aa
    ("मिला", "मिल"),          # milaa / mil   -- vowel-final vs consonant-final
    ("karna", "karne"),       # -aa vs -e
    ("मेरा", "मेरी"),          # meraa / merii
    ("यार", "यहाँ"),          # yaar / yahaan -- r-coda vs nasal vowel
    ("आग", "आज"),            # aag / aaj     -- g vs j coda
    ("दोस्त", "दोस्ती"),        # dost / dosti
    ("गली", "गलत"),          # gali / galat
    ("नज़र", "नज़ारा"),         # nazar / nazaara
    ("जान", "जिन"),          # jaan / jin
    ("रोटी", "रात"),          # roti / raat
    ("पानी", "पन्ने"),         # paani / panne
]

# ── Multisyllabic rhymes: last TWO nuclei onward must match ─────────────────
GOLD_MULTI_POSITIVE = [
    ("दीवाना", "परवाना"),      # deewana / parwana   -> -waanaa? no: -aanaa
    ("zamaana", "nishaana"),  # -aanaa
    ("कहानी", "जवानी"),        # kahaani / jawaani   -> -aanii
    ("mastaani", "deewani"),
    ("ज़िंदगी", "बंदगी"),        # zindagi / bandagi   -> -andagii? at least -agii
    ("अकेला", "मेला"),         # akela / mela        -> -elaa
    ("सहारा", "दुबारा"),        # sahaara / dubaara   -> -aaraa
    ("तन्हाई", "जुदाई"),        # tanhaai / judaai    -> -aaii
    ("भारी", "सारी"),          # bhaari / saari      -> -aarii
    ("सपना", "अपना"),          # sapna / apna        -> -apnaa (medial schwa!)
    ("मरना", "करना"),          # marna / karna       -> -arnaa
]

# ── Final syllable may match, but the 2-nucleus key must NOT ────────────────
GOLD_MULTI_NEGATIVE = [
    ("मरना", "मिलना"),         # marna / milna    -- -arnaa vs -ilnaa
    ("mohabbat", "musibat"),  # -abbat vs -ibat
    ("दीवाना", "दीवानी"),       # deewana / deewani -- -aanaa vs -aanii
    ("कहानी", "पुराना"),        # kahaani / puraana
    ("सोना", "सीना"),          # sonaa / siinaa   -- -onaa vs -iinaa
]

# ── Script equivalence: Devanagari == every corpus romanization ─────────────
# (devanagari, [romanized variants]) -- all must yield identical phonemes.
GOLD_SCRIPT_EQUIV = [
    ("घर", ["ghar"]),
    ("डर", ["dar", "darr"]),
    ("दिल", ["dil"]),
    ("प्यार", ["pyaar", "pyar"]),
    ("यार", ["yaar", "yar"]),
    ("रात", ["raat"]),
    ("बात", ["baat"]),
    ("ज़िंदगी", ["zindagi", "jindagi"]),
    ("दीवाना", ["deewana", "diwana"]),
    ("जहाँ", ["jahaan", "jahan"]),
    ("समझ", ["samajh"]),          # final schwa deletion
    ("सड़क", ["sadak"]),           # nukta flap
    ("अपना", ["apna"]),           # medial schwa deletion
    ("सपना", ["sapna"]),
    ("कमरा", ["kamra"]),          # kamaraa -> kamraa
    ("नौकरी", ["naukri"]),        # naukarii -> naukrii
    ("मुश्किल", ["mushkil"]),      # conjunct
    ("आँख", ["aankh", "ankh"]),   # chandrabindu
    ("हूँ", ["hoon", "hun"]),      # THE spelling-variant collapse case
    ("मैं", ["main"]),
    ("कल", ["kal"]),
    ("काल", ["kaal"]),            # length preserved vs kal
    ("नाम", ["naam"]),
    ("काम", ["kaam"]),
    ("शाम", ["shaam", "sham"]),
    ("पानी", ["paani", "pani"]),
    ("कहानी", ["kahaani", "kahani"]),
    ("जवानी", ["jawaani", "jawani"]),
    ("मोहब्बत", ["mohabbat"]),     # gemination
    ("दर्द", ["dard"]),            # reph / r-conjunct
    ("ख्वाब", ["khwaab", "khwab"]),
    ("किताब", ["kitaab", "kitab"]),
    ("सफ़र", ["safar"]),
    ("नज़र", ["nazar", "najar"]),  # z/j variation
    ("शहर", ["shahar", "sheher", "shehar"]),
    ("लहर", ["lahar", "leher"]),
]
