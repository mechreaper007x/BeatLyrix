# -*- coding: utf-8 -*-
"""
Narci lyrics scraper -- SoulaLyrix (soulalyrix.com) source.

Narci (Sanatan/mythology rap: Ramayana in 10 minutes, Dashavatar,
Mahisasur Mardini) is not reliably on Genius or lyricsmint, so the standard
two-source scraper can't reach him. SoulaLyrix hosts his catalogue on
server-rendered pages tagged /tag/narci/, each with an
"<Title> Lyrics in English" (romanized) section -- exactly the register the
DHH dictionary lacks (Sanskritic/tatsama vocabulary: dashavatar, pawansut,
trilokpati). Records land in corpus/data/narci/ in the same shape as
scrape_corpus.py output; corpus/data stays gitignored as always.

Usage:
    python -m corpus.scrape_narci
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

DATA_DIR = Path(__file__).resolve().parent / "data" / "narci"
TAG_URL = "https://soulalyrix.com/tag/narci/"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/122.0 Safari/537.36")
PAUSE_S = 2.0


def _get(url: str) -> BeautifulSoup:
    resp = requests.get(url, headers={"User-Agent": UA}, timeout=25)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def list_song_pages() -> list[tuple[str, str]]:
    """(title, url) for every Narci song on the tag page (incl. pagination)."""
    pages: list[tuple[str, str]] = []
    url = TAG_URL
    seen_urls: set[str] = set()
    while url:
        soup = _get(url)
        for a in soup.select("h2 a[href], h3 a[href], .entry-title a[href]"):
            href = a.get("href", "")
            title = a.get_text(strip=True)
            if "soulalyrix.com" in href and href not in seen_urls and title:
                seen_urls.add(href)
                pages.append((re.sub(r"\s*Lyrics.*$", "", title, flags=re.I), href))
        nxt = soup.select_one("a.next, .nav-previous a, a[rel=next]")
        url = nxt.get("href") if nxt else None
        if url:
            time.sleep(PAUSE_S)
    return pages


def extract_lyrics(url: str) -> str | None:
    """Pull the romanized ('Lyrics in English') block from a song page."""
    soup = _get(url)
    content = soup.select_one(".entry-content") or soup.body
    if not content:
        return None
    text = content.get_text("\n", strip=True)

    # The lyric body starts after the "<Title> Lyrics in English" heading;
    # some pages only carry the Devanagari section, which is equally usable
    # (the DHH phoneme engine handles both scripts natively).
    m = re.search(r"Lyrics in English\n", text)
    if not m:
        m = re.search(r"Lyrics in Hindi\n", text)
    if not m:
        return None
    body = text[m.end():]
    for stop in ("Lyrics in Hindi", "\nVideo\n", "Music Video", "You May Also",
                 "Related Posts", "Leave a Comment", "FAQ", "Disclaimer"):
        i = body.find(stop)
        if i > 0:
            body = body[:i]
    lines = [ln.strip() for ln in body.splitlines()]
    lines = [ln for ln in lines if ln and len(ln) < 120
             and not re.match(r"^(Singers?|Lyrics|Song|Category|Music|Mix)", ln)]
    return "\n".join(lines).strip() or None


def extract_lyrics_lyricssingh(url: str) -> str | None:
    """Pull the lyric body from a lyricssingh.com song page. Structure: a
    credits block, then '<Title> Lyrics' headings, then the verse text
    running until the video/footer boilerplate."""
    soup = _get(url)
    content = soup.select_one(".entry-content") or soup.body
    if not content:
        return None
    text = content.get_text("\n", strip=True)
    # body starts after the LAST '... Lyrics' heading line near the top
    m = None
    for m in re.finditer(r"^.{0,80}Lyrics\s*$", text[:1500], flags=re.MULTILINE):
        pass
    if not m:
        return None
    body = text[m.end():]
    for stop in ("\nVideo\n", "Music Video", "You May Also", "Related",
                 "Leave a Comment", "Tags", "Share", "Watch Video"):
        i = body.find(stop)
        if i > 0:
            body = body[:i]
    lines = [ln.strip() for ln in body.splitlines()]
    lines = [ln for ln in lines if ln and len(ln) < 120]
    return "\n".join(lines).strip() or None


def list_lyricssingh_songs() -> list[tuple[str, str]]:
    """(title, url) for Narci songs on lyricssingh.com via its search."""
    soup = _get("https://lyricssingh.com/?s=Narci")
    out: list[tuple[str, str]] = []
    for a in soup.select("h2 a[href], h3 a[href], .entry-title a[href]"):
        href, title = a.get("href", ""), a.get_text(strip=True)
        if "lyricssingh.com" in href and "narci" in href.lower():
            out.append((re.sub(r"\s*Lyrics.*$", "", title, flags=re.I), href))
    return out


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    sources = [(list_song_pages, extract_lyrics, "soulalyrix"),
               (list_lyricssingh_songs, extract_lyrics_lyricssingh, "lyricssingh")]
    saved = 0
    for lister, extractor, src in sources:
        try:
            songs = lister()
        except Exception as exc:
            print(f"{src}: listing failed: {type(exc).__name__}: {exc}")
            continue
        print(f"{len(songs)} Narci song pages found on {src}")
        for title, url in songs:
            slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
            out = DATA_DIR / f"{slug}.json"
            if out.exists():
                print(f"  = {title} (already saved)")
                continue
            try:
                lyrics = extractor(url)
            except Exception as exc:
                print(f"  x {title}: {type(exc).__name__}: {exc}")
                continue
            if not lyrics or lyrics.count("\n") < 7:
                print(f"  x {title}: no usable lyric body")
                continue
            rec = {
                "artist": "Narci",
                "title": title,
                "source_url": url,
                "primary_language": "mixed",
                "lyrics": lyrics,
                "line_count": lyrics.count("\n") + 1,
                "char_count": len(lyrics),
            }
            out.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
            saved += 1
            print(f"  + {title} ({rec['line_count']} lines)")
            time.sleep(PAUSE_S)
    print(f"\nsaved {saved} new tracks -> {DATA_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
