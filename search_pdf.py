import re
txt = open(r'C:\Users\Savyasachi Mishra\Desktop\BeatLyrix\raprank-nlp\paper_extract.txt', encoding='utf-8').read()
txt = txt.replace('\x00', '')
out = []
for kw in ['DBSCAN', 'ilhouette', 'psilon', 'ontext', 'ositional', '1 S', 'noise', 'cluster size', 'default']:
    seen = set()
    for m in list(re.finditer(re.escape(kw), txt))[:8]:
        i = m.start()
        snip = txt[max(0, i-400):i+400]
        if snip[:60] in seen:
            continue
        seen.add(snip[:60])
        out.append('=== ' + kw + ' ===\n' + snip + '\n')
open(r'C:\Users\Savyasachi Mishra\Desktop\BeatLyrix\paper_snippets.txt', 'w', encoding='utf-8').write('\n'.join(out))
print('done', len(out))
