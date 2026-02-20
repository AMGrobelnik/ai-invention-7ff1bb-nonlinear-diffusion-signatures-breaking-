"""Assemble final data_out.json, data_out_mini.json, data_out_preview.json."""
import json
from pathlib import Path

WS = Path(__file__).resolve().parent
DS = WS / "temp" / "datasets"

# Load both sources
with open(DS / "brec_records.json") as f:
    brec = json.load(f)
with open(DS / "cfi_records.json") as f:
    cfi = json.load(f)

all_records = brec + cfi
print(f"Total records: {len(all_records)} (BREC={len(brec)}, CFI={len(cfi)})")

# Summary stats
cats = {}
nvt = fv = 0
for r in all_records:
    c = r["metadata"]["category"]
    cats[c] = cats.get(c, 0) + 1
    if not r["metadata"]["graph_1_vertex_transitive"] or not r["metadata"]["graph_2_vertex_transitive"]:
        nvt += 1
    if r["metadata"]["graph_1_features_vary"] or r["metadata"]["graph_2_features_vary"]:
        fv += 1

print("\nCategory breakdown:")
for c, n in sorted(cats.items()):
    print(f"  {c}: {n}")
print(f"\nNon-vertex-transitive (>=1 graph): {nvt}")
print(f"Feature-varying (>=1 graph): {fv}")

# Save full
full_path = DS / "data_out.json"
with open(full_path, "w") as f:
    json.dump(all_records, f, indent=2)
fs = full_path.stat().st_size / (1024*1024)
print(f"\ndata_out.json: {len(all_records)} records, {fs:.1f} MB")

# Mini: 20 Basic + 10 Extension + 10 CFI + 10 Custom_CFI = 50
mini = []
limits = {"Basic": 20, "Extension": 10, "CFI": 10, "Custom_CFI": 10}
cc = {}
for r in all_records:
    c = r["metadata"]["category"]
    lim = limits.get(c, 0)
    if lim > 0 and cc.get(c, 0) < lim:
        mini.append(r)
        cc[c] = cc.get(c, 0) + 1
mp = DS / "data_out_mini.json"
with open(mp, "w") as f:
    json.dump(mini, f, indent=2)
print(f"data_out_mini.json: {len(mini)} records, {mp.stat().st_size/(1024*1024):.2f} MB")

# Preview: 5 records, truncated
prev = []
for r in all_records[:5]:
    pr = json.loads(json.dumps(r))
    for gk in ["graph_1", "graph_2"]:
        g = pr["input"][gk]
        if "adjacency_matrix" in g:
            g["adjacency_matrix"] = g["adjacency_matrix"][:5]
        for fk in g["node_features"]:
            g["node_features"][fk] = g["node_features"][fk][:10]
    prev.append(pr)
pp = DS / "data_out_preview.json"
with open(pp, "w") as f:
    json.dump(prev, f, indent=2)
print(f"data_out_preview.json: {len(prev)} records, {pp.stat().st_size/1024:.1f} KB")

# File size check
if fs > 50:
    print(f"\nWARNING: data_out.json is {fs:.1f}MB (>50MB). Dropping adj matrices for graphs >30 nodes...")
    for r in all_records:
        for gk in ["graph_1", "graph_2"]:
            g = r["input"][gk]
            if g["num_nodes"] > 30 and "adjacency_matrix" in g:
                del g["adjacency_matrix"]
    with open(full_path, "w") as f:
        json.dump(all_records, f, indent=2)
    fs2 = full_path.stat().st_size / (1024*1024)
    print(f"  Reduced to {fs2:.1f} MB")

print("\nDone!")
