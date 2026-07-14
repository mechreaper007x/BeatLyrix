from pathlib import Path
from bs4 import BeautifulSoup
import re

url = "https://genius.com/Raga-jamnapaar-lyrics"

def fetch_html_with_curl(url: str) -> str | None:
    import subprocess
    import tempfile
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as temp_file:
            temp_path = temp_file.name
        cmd = [
            "curl.exe",
            "-s",
            "-o", temp_path,
            "-H", "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
            url
        ]
        res = subprocess.run(cmd, capture_output=True)
        if res.returncode != 0:
            print(f"      x curl failed with code {res.returncode}")
            return None
        p = Path(temp_path)
        if p.exists():
            content = p.read_text(encoding="utf-8", errors="ignore")
            p.unlink()
            return content
        return None
    except Exception as e:
        print(f"      x Exception running curl: {e}")
        return None

html = fetch_html_with_curl(url)
print(f"HTML fetched: {len(html) if html else 0} bytes")

soup = BeautifulSoup(html, "html.parser")
# Check if it has Cloudflare challenge text
if "cloudflare" in html.lower() and "challenge" in html.lower():
    print("WARNING: Cloudflare challenge page returned!")
    print(html[:500])

for s in soup(["script", "style"]):
    s.decompose()

containers = soup.find_all("div", attrs={"data-lyrics-container": "true"})
print(f"Containers found: {len(containers)}")
