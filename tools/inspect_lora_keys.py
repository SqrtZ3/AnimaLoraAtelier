import json, struct, re, sys

p = sys.argv[1]
with open(p, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))
keys = [k for k in hdr if k != "__metadata__"]
print("total keys:", len(keys))
meta = hdr.get("__metadata__", {})
for k in list(meta)[:12]:
    print("META", k, "=", str(meta[k])[:140])
pats = {}
blocks = set()
for k in keys:
    m = re.search(r"blocks[._](\d+)", k)
    if m:
        blocks.add(int(m.group(1)))
    pk = re.sub(r"\d+", "N", k)
    if pk not in pats:
        pats[pk] = [0, k]
    pats[pk][0] += 1
if blocks:
    print("block indices:", min(blocks), "-", max(blocks), "count", len(blocks))
for pk, (c, ex) in sorted(pats.items()):
    print(f"{c:4d}  {ex}  dtype={hdr[ex]['dtype']} shape={hdr[ex]['shape']}")
