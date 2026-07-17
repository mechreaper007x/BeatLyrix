import zlib, re
path = r'C:\Users\Savyasachi Mishra\.claude\projects\c--Users-Savyasachi-Mishra-Desktop-BeatLyrix\1d96bf45-2575-4a93-a1e1-a2de66a2b409\tool-results\webfetch-1784219645657-6sttyf.pdf'
data = open(path, 'rb').read()
texts = []
for m in re.finditer(rb'stream\r?\n(.*?)endstream', data, re.S):
    try:
        texts.append(zlib.decompress(m.group(1)))
    except Exception:
        pass
s = b'\n'.join(texts).decode('latin-1', 'ignore')
pat = re.compile(r'\((?:[^()\\]|\\.)*\)')
txt = ' '.join(c[1:-1] for c in pat.findall(s))
txt = txt.replace('\\(', '(').replace('\\)', ')')
open('paper_extract.txt', 'w', encoding='utf-8').write(txt)
print(len(txt))
