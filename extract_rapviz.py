import zlib, re

pdf = r'C:\Users\Savyasachi Mishra\.claude\projects\c--Users-Savyasachi-Mishra-Desktop-BeatLyrix\1d96bf45-2575-4a93-a1e1-a2de66a2b409\tool-results\webfetch-1784219645045-3lsd5x.pdf'
data = open(pdf, 'rb').read()
streams = re.findall(rb'stream\r?\n(.*?)endstream', data, re.S)
out = []
for s in streams:
    try:
        d = zlib.decompress(s)
        if b'TJ' in d:
            out.append(d)
    except Exception:
        pass
blob = b'\n'.join(out).decode('latin1')
paren = re.compile(r'\((?:[^()\\]|\\.)*\)')

for kw in ['distance', 'positional', 'windo', 'weight', 'default', 'Silhou', 'conte', '1)]TJ', 'minus']:
    for m in re.finditer(re.escape(kw), blob):
        i = m.start()
        seg = blob[max(0, i - 350):i + 450]
        frs = paren.findall(seg)
        txt = ' '.join(f[1:-1] for f in frs)
        print('==', kw, ':', txt)
        print()
