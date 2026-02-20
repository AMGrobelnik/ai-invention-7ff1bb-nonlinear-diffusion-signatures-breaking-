"""Generate additional CFI pairs to reach 20+ total."""
import json, signal, time
from itertools import combinations
from pathlib import Path
import networkx as nx
import numpy as np

WS = Path(__file__).resolve().parent
OUT_DIR = WS / "temp" / "datasets"

class TimeoutErr(Exception): pass
def _handler(s, f): raise TimeoutErr()
signal.signal(signal.SIGALRM, _handler)

def is_vertex_transitive_wl(G):
    nodes = sorted(G.nodes())
    if not nodes: return True
    colors = {v: G.degree(v) for v in nodes}
    cm, nc = {}, max(colors.values()) + 1
    for _ in range(min(len(nodes), 20)):
        new_c = {}
        for v in nodes:
            nb = tuple(sorted(colors[u] for u in G.neighbors(v)))
            k = (colors[v], nb)
            if k not in cm: cm[k] = nc; nc += 1
            new_c[v] = cm[k]
        if new_c == colors: break
        colors = new_c
    return len(set(colors.values())) == 1

def compute_feats(G):
    nodes = sorted(G.nodes()); n = len(nodes)
    d = [G.degree(v) for v in nodes]
    cl = nx.clustering(G); cv = [round(cl[v], 6) for v in nodes]
    if n <= 60:
        bt = nx.betweenness_centrality(G); bv = [round(bt[v], 6) for v in nodes]
    else:
        bv = [0.0]*n
    return {"degree": d, "clustering_coefficient": cv, "betweenness_centrality": bv}

def features_vary(f):
    for k in ["degree","clustering_coefficient"]:
        if len(set(f[k])) > 1: return True
    bv = f["betweenness_centrality"]
    if any(v!=0. for v in bv) and len(set(bv))>1: return True
    return False

def graph_to_dict(G):
    nodes = sorted(G.nodes()); n = len(nodes)
    nm = {v:i for i,v in enumerate(nodes)}
    el = sorted([[nm[u],nm[v]] for u,v in G.edges()])
    r = {"num_nodes": n, "edge_list": el}
    if n <= 50:
        adj = [[0]*n for _ in range(n)]
        for u,v in G.edges():
            i,j = nm[u],nm[v]; adj[i][j]=1; adj[j][i]=1
        r["adjacency_matrix"] = adj
    Gr = nx.relabel_nodes(G, nm)
    r["node_features"] = compute_feats(Gr)
    return r

def build_cfi_pair(base_graph, twist_edge=None):
    base = base_graph.copy()
    nb = sorted(base.nodes())
    ie = {v: sorted(base.edges(v)) for v in nb}
    def eps(d):
        r = []
        for rr in range(0, d+1, 2):
            for c in combinations(range(d), rr): r.append(frozenset(c))
        return r
    def build(tw=None):
        G = nx.Graph(); nc = 0; an = {}; bn = {}
        for v in nb:
            d = base.degree(v); an[v] = {}; bn[v] = {}
            for i in range(d): an[v][i] = nc; G.add_node(nc); nc += 1
            for i in range(d): bn[v][i] = nc; G.add_node(nc); nc += 1
            for S in eps(d):
                mid = nc; G.add_node(nc); nc += 1
                for i in range(d): G.add_edge(mid, an[v][i] if i in S else bn[v][i])
        for u,v in base.edges():
            iu = next(j for j,e in enumerate(ie[u]) if set(e)=={u,v})
            iv = next(j for j,e in enumerate(ie[v]) if set(e)=={u,v})
            t = tw is not None and set(tw)=={u,v}
            if t:
                G.add_edge(an[u][iu], bn[v][iv]); G.add_edge(bn[u][iu], an[v][iv])
            else:
                G.add_edge(an[u][iu], an[v][iv]); G.add_edge(bn[u][iu], bn[v][iv])
        return G
    if twist_edge is None: twist_edge = list(base.edges())[0]
    return build(None), build(twist_edge)

def build_record(G1, G2, pid, bname):
    g1d, g2d = graph_to_dict(G1), graph_to_dict(G2)
    vt1, vt2 = is_vertex_transitive_wl(G1), is_vertex_transitive_wl(G2)
    fv1, fv2 = features_vary(g1d["node_features"]), features_vary(g2d["node_features"])
    r1 = len(set(d for _,d in G1.degree()))==1
    r2 = len(set(d for _,d in G2.degree()))==1
    return {
        "input": {"pair_id": pid, "graph_1": g1d, "graph_2": g2d},
        "output": {"is_isomorphic": False, "is_1wl_equivalent": True, "wl_equivalence_level": "1-WL"},
        "metadata_fold": "test",
        "metadata": {
            "source": "Custom_CFI", "category": "Custom_CFI", "difficulty": "1-WL",
            "base_graph": bname,
            "graph_1_vertex_transitive": vt1, "graph_2_vertex_transitive": vt2,
            "graph_1_features_vary": fv1, "graph_2_features_vary": fv2,
            "graph_1_is_regular": r1, "graph_2_is_regular": r2,
            "num_nodes_g1": G1.number_of_nodes(), "num_nodes_g2": G2.number_of_nodes(),
            "num_edges_g1": G1.number_of_edges(), "num_edges_g2": G2.number_of_edges(),
        },
    }

# Load existing
with open(OUT_DIR / "cfi_records.json") as f:
    cfi_existing = json.load(f)
existing = {r["metadata"]["base_graph"] for r in cfi_existing}
print(f"Existing: {len(cfi_existing)} ({sorted(existing)})")

np.random.seed(42)
new_bases = []
for n in [5,6,8]:
    name = f"path_{n}"
    if name not in existing: new_bases.append((nx.path_graph(n), name))
for n in [10,12]:
    name = f"cycle_{n}"
    if name not in existing: new_bases.append((nx.cycle_graph(n), name))
if "petersen" not in existing:
    new_bases.append((nx.petersen_graph(), "petersen"))
for n in [5,6,7,8]:
    name = f"wheel_{n}"
    if name not in existing: new_bases.append((nx.wheel_graph(n), name))
if "K_4_4" not in existing:
    new_bases.append((nx.complete_bipartite_graph(4,4), "K_4_4"))
for n in [10,12]:
    name = f"random_3reg_{n}"
    if name not in existing:
        try: new_bases.append((nx.random_regular_graph(3, n, seed=42+n), name))
        except: pass
for n in [6,8]:
    name = f"star_{n}"
    if name not in existing: new_bases.append((nx.star_graph(n), name))
for n in [3,4,5]:
    name = f"book_{n}"
    if name not in existing:
        G = nx.Graph()
        for i in range(n): G.add_edges_from([(0,1), (0, 2+i), (1, 2+i)])
        new_bases.append((G, name))
# Ladder graphs
for n in [4,5,6]:
    name = f"ladder_{n}"
    if name not in existing: new_bases.append((nx.ladder_graph(n), name))

print(f"New bases to try: {len(new_bases)}")
new_records = []
for base_G, name in new_bases:
    if not nx.is_connected(base_G):
        print(f"  Skip {name}: disconnected"); continue
    signal.alarm(30)
    try:
        G1, G2 = build_cfi_pair(base_G)
        n = G1.number_of_nodes()
        if n <= 60:
            if nx.is_isomorphic(G1, G2):
                print(f"  Skip {name}: isomorphic"); continue
        rec = build_record(G1, G2, f"custom_cfi_{name}", name)
        new_records.append(rec)
        print(f"  {name}: {n}n OK")
    except TimeoutErr:
        print(f"  Skip {name}: timeout")
    except Exception as e:
        print(f"  Error {name}: {e}")
    finally:
        signal.alarm(0)

all_cfi = cfi_existing + new_records
print(f"\nTotal CFI: {len(all_cfi)}")
with open(OUT_DIR / "cfi_records.json", "w") as f:
    json.dump(all_cfi, f, indent=2)
print("Saved.")
