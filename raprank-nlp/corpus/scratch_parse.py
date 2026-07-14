import sys
from bs4 import BeautifulSoup
from pathlib import Path

html_path = Path(r"C:\Users\Savyasachi Mishra\.gemini\antigravity-ide\brain\4826c60c-6dde-4cbb-a06b-cfe8c25d29ca\.system_generated\steps\688\content.md")

if not html_path.exists():
    print(f"Error: {html_path} does not exist.")
    sys.exit(1)

html_content = html_path.read_text(encoding="utf-8")
soup = BeautifulSoup(html_content, "html.parser")

containers = soup.find_all("div", attrs={"data-lyrics-container": "true"})
if containers:
    text_parts = []
    for container in containers:
        for br in container.find_all("br"):
            br.replace_with("\n")
        text_parts.append(container.get_text())
    lyrics = "\n\n".join(text_parts).strip()
    print("SUCCESSFULLY PARSED REACT CONTAINER:")
    print(lyrics[:500])
else:
    old_container = soup.find("div", class_="lyrics")
    if old_container:
        print("SUCCESSFULLY PARSED OLD CONTAINER:")
        print(old_container.get_text()[:500])
    else:
        print("FAILED TO PARSE LYRICS CONTAINER.")
