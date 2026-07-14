"""
Two-source lyrics scraper for the RapRank calibration corpus.

Neither source alone is sufficient from a server/datacenter environment:

  * Genius API (api.genius.com) — authenticated, reachable, and gives the
    authoritative, complete song catalogue per artist. But it never returns
    lyric *text*, and the genius.com HTML pages that host the text are
    Cloudflare-blocked from datacenter IPs (403).

  * lyricsmint.com — serves clean Romanized-Hinglish lyric text on its song
    pages (ideal for this NLP), but its search / index pages are JS-rendered
    so songs cannot be enumerated from it.

So we combine them: Genius API enumerates every song title per artist, and we
map each title to a lyricsmint song URL (`/<artist-slug>/<song-slug>`), trying
a few slug variants and auto-discovering the artist slug. Lyric text is stored
locally only for offline NLP calibration; corpus/data/ is git-ignored.

Usage:
    GENIUS_ACCESS_TOKEN=xxxx python -m corpus.scrape_corpus
    GENIUS_ACCESS_TOKEN=xxxx python -m corpus.scrape_corpus --artist "KR$NA"
    GENIUS_ACCESS_TOKEN=xxxx python -m corpus.scrape_corpus --resume --max 60

Must run with the shell sandbox disabled (network required).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

try:
    from corpus.artists import Artist, unique_artists
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from corpus.artists import Artist, unique_artists

API_BASE = "https://api.genius.com"
DATA_DIR = Path(__file__).resolve().parent / "data"
BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
)
PAUSE_S = 0.6


# ── slugging ──────────────────────────────────────────────────────────────
def title_slug(title: str) -> str:
    t = title.lower()
    t = re.sub(r"\(.*?\)", "", t)                 # drop "(feat ...)", "(Romanized)"
    t = re.sub(r"\bfeat\.?\b.*$", "", t)          # drop trailing "feat ..."
    t = t.replace("&", "and").replace("$", "s")
    t = re.sub(r"[^a-z0-9\s-]", "", t)
    t = re.sub(r"[\s-]+", "-", t).strip("-")
    return t


def slug_variants(title: str) -> list[str]:
    base = title_slug(title)
    variants = [base]
    # keep the leading number stripped / kept ("machayenge-4")
    no_trailing_num = re.sub(r"-\d+$", "", base)
    if no_trailing_num != base:
        variants.append(no_trailing_num)
    # collapse "and" back out (some slugs drop the ampersand entirely)
    v2 = base.replace("-and-", "-")
    if v2 != base:
        variants.append(v2)
    # first token only (very short titles sometimes truncate)
    seen, out = set(), []
    for v in variants:
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def artist_slug_candidates(name: str) -> list[str]:
    base = name.lower().replace("$", "s")
    base = re.sub(r"[^a-z0-9\s]", "", base)
    base = re.sub(r"\s+", "-", base).strip("-")
    cands = [base, base.replace("-", "")]
    # KR$NA -> kr-na (lyricsmint keeps the $ position as a hyphen)
    if "$" in name:
        cands.append(name.lower().replace("$", "-"))
    return list(dict.fromkeys(c for c in cands if c))


# ── Genius API ─────────────────────────────────────────────────────────────
class Genius:
    def __init__(self, token: str):
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {token}", "User-Agent": "curl/8.0"})

    def _get(self, path: str, params: dict | None = None) -> dict:
        r = self.s.get(f"{API_BASE}{path}", params=params or {}, timeout=25)
        r.raise_for_status()
        return r.json()

    def resolve(self, artist: Artist) -> tuple[int | None, str | None]:
        if artist.genius_artist_id:
            return artist.genius_artist_id, artist.name
        want = _norm(artist.name)
        for term in artist.search_terms:
            try:
                hits = self._get("/search", {"q": term})["response"]["hits"]
            except requests.RequestException as e:
                print(f"    ! search '{term}' failed: {e}")
                continue
            for h in hits:
                pa = h["result"]["primary_artist"]
                if pa.get("id") and _name_matches(want, pa["name"]):
                    return pa["id"], pa["name"]
        # loose fallback: first primary artist of first term
        for term in artist.search_terms:
            try:
                hits = self._get("/search", {"q": term})["response"]["hits"]
            except requests.RequestException:
                continue
            if hits:
                pa = hits[0]["result"]["primary_artist"]
                print(f"    ~ loose match '{artist.name}' -> '{pa['name']}' (verify)")
                return pa["id"], pa["name"]
        return None, None

    def songs(self, artist_id: int, max_songs: int | None) -> list[str]:
        titles: list[str] = []
        page = 1
        while True:
            data = self._get(f"/artists/{artist_id}/songs",
                             {"per_page": 50, "page": page, "sort": "popularity"})
            resp = data["response"]
            for s in resp["songs"]:
                if s.get("primary_artist", {}).get("id") == artist_id:
                    titles.append(s["title"])
            if max_songs and len(titles) >= max_songs:
                return titles[:max_songs]
            if not resp.get("next_page"):
                break
            page = resp["next_page"]
            time.sleep(0.25)
        return titles


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower().replace("$", "s"))


def _name_matches(want_norm: str, got: str) -> bool:
    g = _norm(got)
    return bool(want_norm) and (want_norm == g or want_norm in g or g in want_norm)


# ── lyricsmint text ─────────────────────────────────────────────────────────
def parse_lyricsmint(html: str) -> str | None:
    """Lyric text lives in the div carrying the most <br> tags on the page."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "form"]):
        tag.decompose()
    best, best_brs = None, 0
    for d in soup.find_all("div"):
        brs = len(d.find_all("br"))
        if brs > best_brs:
            best, best_brs = d, brs
    if not best or best_brs < 4:
        return None
    for br in best.find_all("br"):
        br.replace_with("\n")
    text = best.get_text()
    text = _strip_cruft(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(l.rstrip() for l in text.splitlines()).strip()
    return _dedup_blocks(text) if text else None


# lyricsmint wraps the lyric block with a share widget + footer we must drop
_CRUFT_LINE = re.compile(
    r"^\s*(print|share|whatsapp|x \(twitter\)|facebook|copy|more options\.*|"
    r"select any lyrics text.*|.*\blyrics\b\s*$)\s*$",
    re.IGNORECASE,
)
_START_MARKER = "select any lyrics text"
_END_MARKERS = ("written by", "explore more songs", "related songs",
                "song info", "top 10 songs", "lyricsmint recommends",
                "more options", "you might also like", "read more",
                "share lyrics image")
_END_RE = re.compile(r'^\s*".*"\s*video\s*$', re.IGNORECASE)


def _strip_cruft(text: str) -> str:
    lines = text.splitlines()
    # Cut everything up to and including the share-widget start marker, if present
    start = 0
    for i, l in enumerate(lines):
        if _START_MARKER in l.lower():
            start = i + 1
            break
    lines = lines[start:]
    # Trim trailing footer cruft (first line that is clearly a footer marker)
    end = len(lines)
    for i, l in enumerate(lines):
        low = l.lower().strip()
        if _END_RE.match(l) or any(m in low for m in _END_MARKERS):
            end = i
            break
    lines = lines[:end]
    # Drop any residual widget lines and leading blanks
    lines = [l for l in lines if not _CRUFT_LINE.match(l)]
    while lines and not lines[0].strip():
        lines.pop(0)
    return "\n".join(lines)


def _dedup_blocks(text: str) -> str:
    """lyricsmint often prints the hook block twice back-to-back — drop the
    immediate duplicate of the opening block."""
    lines = text.splitlines()
    half = len(lines) // 2
    if half > 4 and lines[:half] == lines[half:half * 2]:
        return "\n".join(lines[half:]).strip()
    return text


def fetch_lyrics(session: requests.Session, artist_slug: str, title: str) -> tuple[str | None, str | None]:
    for sv in slug_variants(title):
        url = f"https://lyricsmint.com/{artist_slug}/{sv}"
        try:
            r = session.get(url, timeout=15)
        except requests.RequestException:
            continue
        if r.status_code == 200:
            lyr = parse_lyricsmint(r.text)
            if lyr and len(lyr) >= 60:
                return lyr, url
        time.sleep(0.15)
    return None, None


def discover_artist_slug(session: requests.Session, name: str, sample_titles: list[str]) -> str | None:
    """Pick the lyricsmint artist slug that resolves the most sample songs."""
    best, best_hits = None, 0
    for cand in artist_slug_candidates(name):
        hits = 0
        for t in sample_titles[:6]:
            lyr, _ = fetch_lyrics(session, cand, t)
            if lyr:
                hits += 1
        if hits > best_hits:
            best, best_hits = cand, hits
        if best_hits >= 3:  # good enough, stop probing candidates
            break
    return best if best_hits > 0 else None


# ── per-artist driver ────────────────────────────────────────────────────────
def scrape_artist(genius: Genius, session: requests.Session, artist: Artist,
                  max_songs: int | None, resume: bool) -> dict:
    print(f"\n=== {artist.name} ===")
    aid, matched = genius.resolve(artist)
    if not aid:
        print("    x could not resolve on Genius")
        return {"artist": artist.name, "resolved": False, "saved": 0}
    print(f"    Genius id={aid} ({matched})")

    titles = genius.songs(aid, max_songs)
    print(f"    {len(titles)} songs enumerated on Genius")
    if not titles:
        return {"artist": artist.name, "resolved": True, "saved": 0, "enumerated": 0}

    lm_slug = artist.lyricsmint_slug or discover_artist_slug(session, artist.name, titles)
    if not lm_slug:
        print(f"    x no lyricsmint artist slug found for {artist.name}")
        return {"artist": artist.name, "resolved": True, "saved": 0,
                "enumerated": len(titles), "lyrics_source": "none"}
    print(f"    lyricsmint slug -> '{lm_slug}'")

    out_dir = DATA_DIR / re.sub(r"[^a-z0-9]+", "-", artist.name.lower()).strip("-")
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = skipped = missed = 0
    for i, title in enumerate(titles, 1):
        fname = f"{title_slug(title) or 'untitled'}.json"
        path = out_dir / fname
        if resume and path.exists():
            skipped += 1
            continue
        lyr, url = fetch_lyrics(session, lm_slug, title)
        if not lyr:
            missed += 1
            continue
        rec = {
            "artist": artist.name,
            "genius_artist_id": aid,
            "title": title,
            "lyricsmint_url": url,
            "primary_language": artist.primary_language,
            "lyrics": lyr,
            "line_count": sum(1 for l in lyr.splitlines() if l.strip()),
            "char_count": len(lyr),
        }
        path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        saved += 1
        if saved % 10 == 0 or i == len(titles):
            print(f"    [{i}/{len(titles)}] saved={saved} missed={missed}")
        time.sleep(PAUSE_S)

    print(f"    done: {saved} saved, {skipped} skipped, {missed} not-on-lyricsmint")
    return {"artist": artist.name, "resolved": True, "enumerated": len(titles),
            "saved": saved, "skipped": skipped, "missed": missed, "lyricsmint_slug": lm_slug}


def main() -> int:
    ap = argparse.ArgumentParser(description="Scrape Genius(catalogue)+lyricsmint(text) corpus")
    ap.add_argument("--artist", help="only this artist (by display name)")
    ap.add_argument("--max", type=int, default=None, help="cap songs per artist")
    ap.add_argument("--resume", action="store_true", help="skip already-saved tracks")
    args = ap.parse_args()

    token = os.getenv("GENIUS_ACCESS_TOKEN") or os.getenv("GENIUS_API_TOKEN")
    if not token:
        print("ERROR: set GENIUS_ACCESS_TOKEN", file=sys.stderr)
        return 2

    genius = Genius(token)
    session = requests.Session()
    session.headers.update({"User-Agent": BROWSER_UA})

    targets = unique_artists()
    if args.artist:
        targets = [a for a in targets if a.name.lower() == args.artist.lower()]
        if not targets:
            print(f"no such artist: {args.artist}", file=sys.stderr)
            return 2

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    summary = [scrape_artist(genius, session, a, args.max, args.resume) for a in targets]
    (DATA_DIR / "_scrape_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    total = sum(s.get("saved", 0) for s in summary)
    print(f"\n==== TOTAL SAVED: {total} tracks across {len(targets)} artists ====")
    for s in summary:
        print(f"  {s['artist']:14} saved={s.get('saved',0):3} "
              f"enumerated={s.get('enumerated',0):3} missed={s.get('missed',0):3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
