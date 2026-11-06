from collections import Counter
from pathlib import Path

from src.hospital_rag.chunking import chunk_sops
from src.hospital_rag.parsing import parse_pdf

pages = parse_pdf(Path("data/raw/sample_sops.pdf"))
chunks = chunk_sops(pages, "sops")
lengths = sorted(len(c.text) for c in chunks)

print("chunks:", len(chunks))
print("min / median / max chars:", lengths[0], lengths[len(lengths) // 2], lengths[-1])

print("\nShortest 5:")
for c in sorted(chunks, key=lambda c: len(c.text))[:5]:
    print(len(c.text), c.metadata["sop_id"], c.metadata["section"])

print("\nLongest 5:")
for c in sorted(chunks, key=lambda c: len(c.text), reverse=True)[:5]:
    print(len(c.text), c.metadata["sop_id"], c.metadata["section"])

line_counts = Counter(
    line.strip() for p in pages for line in p.text.splitlines() if line.strip()
)
print("\nLines repeated 3+ times (possible headers/footers):")
for line, count in line_counts.most_common():
    if count >= 3:
        print(count, line[:80])