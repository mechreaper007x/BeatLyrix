import re
txt = open(r'C:\Users\Savyasachi Mishra\Desktop\BeatLyrix\raprank-nlp\paper_extract.txt', encoding='utf-8').read()
txt = txt.replace('\x00', '-')
out = []
for kw in ['ositi', 'onte xt', 'context', 'windo w', 'window', 'weight', 'Silhou', 'Expert', 'e xpert set', 'options', 'penal']:
    seen = set()
    for m in list(re.finditer(kw, txt, re.I))[:10]:
        i = m.start()
        snip = txt[max(0, i-350):i+450]
        key = snip[:80]
        if key in seen:
            continue
        seen.add(key)
        out.append('=== ' + kw + ' ===\n' + snip + '\n')
open(r'C:\Users\Savyasachi Mishra\Desktop\BeatLyrix\paper_snippets2.txt', 'w', encoding='utf-8').write('\n'.join(out))
print('done', len(out))
