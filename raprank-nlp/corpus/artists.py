"""
Seed list of target artists for the RapRank NLP calibration corpus.

Each entry maps a display name to the search term(s) used against the Genius
search API and, optionally, a hard-pinned Genius artist id when the search is
ambiguous (e.g. "Calm" and "Karma" are common English words that collide with
unrelated artists).

`expected_profile` captures the qualitative character of the artist's catalogue.
It is NOT used by the scraper — it documents why the artist is in the corpus and
is consumed by the calibration tests (tests/test_calibration.py) as a coarse
ground-truth for relative-ordering assertions (e.g. a dense multisyllabic
lyricist should out-score a commercial hook-driven one on the rhyme metric).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Artist:
    name: str                       # canonical display name
    search_terms: tuple[str, ...]   # queries tried against Genius search
    genius_artist_id: int | None = None  # pin when search is ambiguous
    lyricsmint_slug: str | None = None   # pin lyricsmint artist slug when auto-discovery is unreliable
    primary_language: str = "hi"    # dominant lyric language: hi | en | mixed
    expected_profile: dict = field(default_factory=dict)


# Ordered roughly by how technical/dense the catalogue skews. `expected_profile`
# fields are coarse 0-1 priors, NOT targets — the tests assert *relative* order
# between artists, never absolute values.
ARTISTS: tuple[Artist, ...] = (
    Artist(
        name="KR$NA",
        search_terms=("KR$NA", "Krsna rapper", "KRSNA"),
        lyricsmint_slug="kr-na",
        primary_language="mixed",
        expected_profile={"multisyllabic": 0.9, "wordplay": 0.95, "commercial": 0.2},
    ),
    Artist(
        name="Seedhe Maut",
        search_terms=("Seedhe Maut",),
        lyricsmint_slug="seedhe-maut",
        primary_language="mixed",
        expected_profile={"multisyllabic": 0.9, "wordplay": 0.9, "commercial": 0.15},
    ),
    Artist(
        name="Encore ABJ",
        search_terms=("Encore ABJ", "Seedhe Maut Encore"),
        primary_language="mixed",
        expected_profile={"multisyllabic": 0.85, "wordplay": 0.85, "commercial": 0.2},
    ),
    Artist(
        name="Calm",
        search_terms=("Calm Seedhe Maut", "Seedhe Maut Calm"),
        primary_language="mixed",
        expected_profile={"multisyllabic": 0.85, "wordplay": 0.8, "commercial": 0.2},
    ),
    Artist(
        name="Karma",
        search_terms=("Karma rapper India", "Karma Vivek Arora"),
        primary_language="mixed",
        expected_profile={"multisyllabic": 0.8, "wordplay": 0.85, "commercial": 0.25},
    ),
    Artist(
        name="EPR",
        search_terms=("EPR Iyer", "EPR rapper"),
        genius_artist_id=2099093,
        lyricsmint_slug="epr-iyer",
        primary_language="mixed",
        expected_profile={"multisyllabic": 0.85, "wordplay": 0.9, "commercial": 0.2},
    ),
    Artist(
        name="Dhanji",
        search_terms=("Dhanji rapper", "Dhanji"),
        primary_language="mixed",
        expected_profile={"multisyllabic": 0.8, "wordplay": 0.85, "commercial": 0.25},
    ),
    Artist(
        name="Brodha V",
        search_terms=("Brodha V",),
        primary_language="mixed",
        expected_profile={"multisyllabic": 0.75, "wordplay": 0.8, "commercial": 0.35},
    ),
    Artist(
        name="Hanumankind",
        search_terms=("Hanumankind",),
        primary_language="en",
        expected_profile={"multisyllabic": 0.7, "wordplay": 0.7, "commercial": 0.4},
    ),
    Artist(
        name="Yashraj",
        search_terms=("Yashraj rapper", "Yashraj Mukhate"),  # disambiguated at scrape time
        primary_language="hi",
        expected_profile={"multisyllabic": 0.7, "wordplay": 0.75, "commercial": 0.35},
    ),
    Artist(
        name="Raftaar",
        search_terms=("Raftaar",),
        primary_language="mixed",
        expected_profile={"multisyllabic": 0.8, "wordplay": 0.65, "commercial": 0.6},
    ),
    Artist(
        name="Muhfaad",
        search_terms=("Muhfaad",),
        primary_language="hi",
        expected_profile={"multisyllabic": 0.65, "wordplay": 0.6, "commercial": 0.5},
    ),
    Artist(
        name="JTrix",
        search_terms=("JTrix", "J Trix rapper"),
        lyricsmint_slug="j-trix",
        primary_language="mixed",
        expected_profile={"multisyllabic": 0.6, "wordplay": 0.6, "commercial": 0.5},
    ),
    Artist(
        name="Emiway Bantai",
        search_terms=("Emiway Bantai", "Emiway"),
        lyricsmint_slug="emiway-bantai",
        primary_language="mixed",
        expected_profile={"multisyllabic": 0.4, "wordplay": 0.4, "commercial": 0.8},
    ),
    Artist(
        name="CarryMinati",
        search_terms=("CarryMinati",),
        lyricsmint_slug="carryminati",
        primary_language="hi",
        expected_profile={"multisyllabic": 0.25, "wordplay": 0.25, "commercial": 0.9},
    ),
    Artist(
        name="King",
        search_terms=("King rapper India", "King Rocco"),
        genius_artist_id=2047810,
        lyricsmint_slug="king",
        primary_language="hi",
        expected_profile={"multisyllabic": 0.35, "wordplay": 0.35, "commercial": 0.85},
    ),
    Artist(
        name="Naam Sujal",
        search_terms=("Naam Sujal",),
        genius_artist_id=2780157,
        primary_language="mixed",
        expected_profile={"multisyllabic": 0.75, "wordplay": 0.7, "commercial": 0.2},
    ),
    Artist(
        name="Vichaar",
        search_terms=("Vichaar", "Vichaar rapper"),
        genius_artist_id=3815901,
        primary_language="mixed",
        expected_profile={"multisyllabic": 0.7, "wordplay": 0.7, "commercial": 0.2},
    ),
    Artist(
        name="Lil Bhatia",
        search_terms=("Lil Bhavi", "Lil Bhatia"),
        genius_artist_id=3495224,
        primary_language="mixed",
        expected_profile={"multisyllabic": 0.75, "wordplay": 0.75, "commercial": 0.15},
    ),
    Artist(
        name="Yungsta",
        search_terms=("Yungsta", "Yungsta rapper"),
        genius_artist_id=174215,
        primary_language="mixed",
        expected_profile={"multisyllabic": 0.8, "wordplay": 0.8, "commercial": 0.25},
    ),
    Artist(
        name="Raga",
        search_terms=("Raga", "Raga rapper", "Ravi Mishra"),
        genius_artist_id=1050178,
        primary_language="mixed",
        expected_profile={"multisyllabic": 0.6, "wordplay": 0.5, "commercial": 0.6},
    ),
    Artist(
        name="Panther",
        search_terms=("Panther (IND)", "Panther rapper"),
        genius_artist_id=3354305,
        primary_language="mixed",
        expected_profile={"multisyllabic": 0.7, "wordplay": 0.6, "commercial": 0.5},
    ),
    Artist(
        name="Paradox",
        search_terms=("Paradox", "Paradox rapper"),
        genius_artist_id=3509765,
        primary_language="mixed",
        expected_profile={"multisyllabic": 0.75, "wordplay": 0.7, "commercial": 0.6},
    ),
    Artist(
        name="Tsumyoki",
        search_terms=("Tsumyoki", "Tsumyoki rapper"),
        genius_artist_id=1691777,
        primary_language="mixed",
        expected_profile={"multisyllabic": 0.55, "wordplay": 0.55, "commercial": 0.7},
    ),
    # ── Expansion wave 2 (Jul 2026): grow the DHH pronunciation dictionary
    # toward 30k entries. Same pipeline, same local-only lyric storage; only
    # derived phonetic facts are ever committed.
    Artist(
        name="DIVINE",
        search_terms=("DIVINE", "Divine rapper Gully Gang"),
        primary_language="mixed",
        expected_profile={"multisyllabic": 0.75, "wordplay": 0.7, "commercial": 0.6},
    ),
    Artist(
        name="MC Stan",
        search_terms=("MC Stan", "MC STAN"),
        primary_language="mixed",
        expected_profile={"multisyllabic": 0.5, "wordplay": 0.5, "commercial": 0.8},
    ),
    Artist(
        name="Prabh Deep",
        search_terms=("Prabh Deep",),
        primary_language="mixed",
        expected_profile={"multisyllabic": 0.8, "wordplay": 0.75, "commercial": 0.3},
    ),
    Artist(
        name="Ikka",
        search_terms=("Ikka", "Ikka Singh"),
        primary_language="mixed",
        expected_profile={"multisyllabic": 0.7, "wordplay": 0.7, "commercial": 0.5},
    ),
    Artist(
        name="Dino James",
        search_terms=("Dino James",),
        primary_language="mixed",
        expected_profile={"multisyllabic": 0.45, "wordplay": 0.5, "commercial": 0.75},
    ),
    Artist(
        name="Talha Anjum",
        search_terms=("Talha Anjum", "Young Stunners Talha"),
        primary_language="mixed",
        expected_profile={"multisyllabic": 0.75, "wordplay": 0.75, "commercial": 0.5},
    ),
    Artist(
        name="Talhah Yunus",
        search_terms=("Talhah Yunus", "Young Stunners Yunus"),
        primary_language="mixed",
        expected_profile={"multisyllabic": 0.75, "wordplay": 0.7, "commercial": 0.45},
    ),
    Artist(
        name="Kaam Bhaari",
        search_terms=("Kaam Bhaari",),
        primary_language="mixed",
        expected_profile={"multisyllabic": 0.7, "wordplay": 0.7, "commercial": 0.45},
    ),
    Artist(
        name="MC Altaf",
        search_terms=("MC Altaf",),
        primary_language="mixed",
        expected_profile={"multisyllabic": 0.65, "wordplay": 0.6, "commercial": 0.5},
    ),
    Artist(
        name="Badshah",
        search_terms=("Badshah", "Badshah rapper"),
        primary_language="mixed",
        expected_profile={"multisyllabic": 0.3, "wordplay": 0.35, "commercial": 0.95},
    ),
    Artist(
        name="Yo Yo Honey Singh",
        search_terms=("Yo Yo Honey Singh", "Honey Singh"),
        primary_language="mixed",
        expected_profile={"multisyllabic": 0.3, "wordplay": 0.35, "commercial": 0.95},
    ),
    Artist(
        name="Bali",
        search_terms=("Bali rapper", "Bali (IND)"),
        primary_language="mixed",
        expected_profile={"multisyllabic": 0.6, "wordplay": 0.55, "commercial": 0.55},
    ),
    Artist(
        name="Fotty Seven",
        search_terms=("Fotty Seven",),
        primary_language="mixed",
        expected_profile={"multisyllabic": 0.6, "wordplay": 0.6, "commercial": 0.6},
    ),
    Artist(
        name="Shah Rule",
        search_terms=("Shah Rule",),
        primary_language="mixed",
        expected_profile={"multisyllabic": 0.6, "wordplay": 0.6, "commercial": 0.55},
    ),
    # ── Expansion wave 3 (Jul 2026): lyricism-first rappers for the DHH
    # dictionary -- dense multisyllabic Hindi vocabulary is the selection
    # criterion, not popularity.
    Artist(
        name="Narci",
        search_terms=("Narci", "Narci rapper"),
        primary_language="mixed",
        expected_profile={"multisyllabic": 0.9, "wordplay": 0.85, "commercial": 0.3},
    ),
    Artist(
        name="Rawal",
        search_terms=("Rawal", "Rawal rapper"),
        primary_language="mixed",
        expected_profile={"multisyllabic": 0.7, "wordplay": 0.7, "commercial": 0.45},
    ),
    Artist(
        name="Sikander Kahlon",
        search_terms=("Sikander Kahlon",),
        primary_language="mixed",
        expected_profile={"multisyllabic": 0.85, "wordplay": 0.8, "commercial": 0.35},
    ),
    Artist(
        name="Smoke",
        search_terms=("Smoke rapper India", "Smoke DHH"),
        primary_language="mixed",
        expected_profile={"multisyllabic": 0.75, "wordplay": 0.7, "commercial": 0.4},
    ),
    Artist(
        name="Bella",
        search_terms=("Bella rapper", "Bella DHH"),
        primary_language="mixed",
        expected_profile={"multisyllabic": 0.7, "wordplay": 0.65, "commercial": 0.5},
    ),
    Artist(
        name="Kr@ntinaari",
        search_terms=("Krantinaari", "Kr@ntinaari"),
        primary_language="mixed",
        expected_profile={"multisyllabic": 0.7, "wordplay": 0.7, "commercial": 0.35},
    ),
    Artist(
        name="Ab 17",
        search_terms=("Ab 17", "AB17 rapper"),
        primary_language="mixed",
        expected_profile={"multisyllabic": 0.75, "wordplay": 0.7, "commercial": 0.4},
    ),
    Artist(
        name="Full Power",
        search_terms=("Full Power rapper", "Fullpower DHH"),
        primary_language="mixed",
        expected_profile={"multisyllabic": 0.7, "wordplay": 0.65, "commercial": 0.4},
    ),
    Artist(
        name="Panda",
        search_terms=("Panda rapper India", "Panda DHH"),
        primary_language="mixed",
        expected_profile={"multisyllabic": 0.65, "wordplay": 0.65, "commercial": 0.45},
    ),
    Artist(
        name="Shaikhspeare",
        search_terms=("Shaikhspeare",),
        primary_language="mixed",
        expected_profile={"multisyllabic": 0.8, "wordplay": 0.85, "commercial": 0.3},
    ),
    Artist(
        name="Poetik Justis",
        search_terms=("Poetik Justis",),
        primary_language="mixed",
        expected_profile={"multisyllabic": 0.75, "wordplay": 0.75, "commercial": 0.35},
    ),
    Artist(
        name="Enkore",
        search_terms=("Enkore", "Enkore rapper"),
        primary_language="mixed",
        expected_profile={"multisyllabic": 0.75, "wordplay": 0.75, "commercial": 0.35},
    ),
)



def unique_artists() -> list[Artist]:
    """De-duplicate by name, keeping the first (richest) definition."""
    seen: set[str] = set()
    out: list[Artist] = []
    for a in ARTISTS:
        if a.name in seen:
            continue
        seen.add(a.name)
        out.append(a)
    return out
