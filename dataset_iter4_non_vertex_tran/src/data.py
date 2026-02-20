# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Convert graph pair dataset into exp_sel_data_out.json schema.

Reads temp/datasets/data_out.json (429 graph pairs from BREC + custom CFI)
and produces full_data_out.json with the 2 selected datasets:
  1. BREC_Basic — 60 non-vertex-transitive, non-regular pairs (1-WL)
  2. Custom_CFI — 29 custom CFI pairs from diverse base graphs

These 89 pairs were selected as the best combination for evaluating NDS
expressiveness: BREC_Basic provides non-regular, non-vertex-transitive
pairs where scalar node features vary, and Custom_CFI provides diverse
CFI-constructed pairs that are provably hard for k-WL tests.
"""

import json
import resource
from pathlib import Path

# Resource limits (14GB RAM, 1h CPU)
resource.setrlimit(resource.RLIMIT_AS, (14 * 1024**3, 14 * 1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))

WS = Path(__file__).resolve().parent
INPUT_FILE = WS / "temp" / "datasets" / "data_out.json"
OUTPUT_FILE = WS / "full_data_out.json"

# Dataset grouping rules — only selected datasets
DATASET_MAP = {
    "Basic": "BREC_Basic",
    "Custom_CFI": "Custom_CFI",
}

# Categories to include (skip all others)
SELECTED_DATASETS = {"BREC_Basic", "Custom_CFI"}


def record_to_example(record: dict, idx: int) -> dict:
    """Convert one graph pair record into schema-compliant example."""
    inp = record["input"]
    out = record["output"]
    meta = record["metadata"]

    # Build input: JSON string of graph pair data
    input_data = {
        "pair_id": inp["pair_id"],
        "graph_1": {
            "num_nodes": inp["graph_1"]["num_nodes"],
            "edge_list": inp["graph_1"]["edge_list"],
            "node_features": inp["graph_1"]["node_features"],
        },
        "graph_2": {
            "num_nodes": inp["graph_2"]["num_nodes"],
            "edge_list": inp["graph_2"]["edge_list"],
            "node_features": inp["graph_2"]["node_features"],
        },
    }
    # Include adjacency matrices only for small graphs (<=50 nodes)
    if "adjacency_matrix" in inp["graph_1"]:
        input_data["graph_1"]["adjacency_matrix"] = inp["graph_1"]["adjacency_matrix"]
    if "adjacency_matrix" in inp["graph_2"]:
        input_data["graph_2"]["adjacency_matrix"] = inp["graph_2"]["adjacency_matrix"]

    # Build output: JSON string of expected properties
    output_data = {
        "is_isomorphic": out["is_isomorphic"],
        "is_1wl_equivalent": out["is_1wl_equivalent"],
        "wl_equivalence_level": out["wl_equivalence_level"],
    }

    example = {
        "input": json.dumps(input_data, separators=(",", ":")),
        "output": json.dumps(output_data, separators=(",", ":")),
        "metadata_fold": 0,  # all test data
        "metadata_pair_id": inp["pair_id"],
        "metadata_category": meta["category"],
        "metadata_difficulty": meta["difficulty"],
        "metadata_source": meta["source"],
        "metadata_task_type": "graph_pair_classification",
        "metadata_graph_1_vertex_transitive": meta["graph_1_vertex_transitive"],
        "metadata_graph_2_vertex_transitive": meta["graph_2_vertex_transitive"],
        "metadata_graph_1_features_vary": meta["graph_1_features_vary"],
        "metadata_graph_2_features_vary": meta["graph_2_features_vary"],
        "metadata_graph_1_is_regular": meta["graph_1_is_regular"],
        "metadata_graph_2_is_regular": meta["graph_2_is_regular"],
        "metadata_num_nodes_g1": meta["num_nodes_g1"],
        "metadata_num_nodes_g2": meta["num_nodes_g2"],
        "metadata_num_edges_g1": meta["num_edges_g1"],
        "metadata_num_edges_g2": meta["num_edges_g2"],
        "metadata_row_index": idx,
    }

    # Add optional metadata
    if "brec_pair_index" in meta:
        example["metadata_brec_pair_index"] = meta["brec_pair_index"]
    if "base_graph" in meta:
        example["metadata_base_graph"] = meta["base_graph"]
    if "graph6_g1" in meta:
        example["metadata_graph6_g1"] = meta["graph6_g1"]
    if "graph6_g2" in meta:
        example["metadata_graph6_g2"] = meta["graph6_g2"]

    return example


def main():
    print(f"Loading {INPUT_FILE}...")
    with open(INPUT_FILE) as f:
        records = json.load(f)
    print(f"Loaded {len(records)} records")

    # Group by dataset — only include selected datasets
    groups = {}
    skipped = 0
    for idx, record in enumerate(records):
        category = record["metadata"]["category"]
        dataset_name = DATASET_MAP.get(category)
        if dataset_name is None or dataset_name not in SELECTED_DATASETS:
            skipped += 1
            continue
        if dataset_name not in groups:
            groups[dataset_name] = []
        groups[dataset_name].append(record_to_example(record, idx))
    print(f"Skipped {skipped} records from non-selected categories")

    # Build output in consistent order
    dataset_order = ["BREC_Basic", "Custom_CFI"]
    datasets = []
    for ds_name in dataset_order:
        if ds_name in groups:
            examples = groups[ds_name]
            datasets.append({"dataset": ds_name, "examples": examples})
            print(f"  {ds_name}: {len(examples)} examples")

    output = {"datasets": datasets}

    print(f"\nWriting {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    total = sum(len(ds["examples"]) for ds in datasets)
    print(f"Wrote {total} examples across {len(datasets)} datasets ({size_mb:.1f} MB)")
    print("Done!")


if __name__ == "__main__":
    main()
