# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""
Convert the constructed 1-WL Equivalent Graph Pairs dataset (data_out.json)
into the exp_sel_data_out.json schema format (full_data_out.json).

Each graph pair becomes one example. The input is a JSON-encoded representation
of both graphs (adjacency matrices, edge lists, eigenvalues, etc.). The output
is the ground-truth label: whether the pair is isomorphic or not.

Single dataset: "1wl_equivalent_graph_pairs" — all 9 pairs across 4 families:
  - srg(16,6,2,2): Shrikhande vs Rook's graph (1 pair)
  - srg(25,12,5,6): Paulus graphs from Spence DB (2 pairs)
  - srg(26,10,3,4): Paulus graphs from Spence DB (1 pair)
  - CSL(41,R): Circular Skip Links graphs (5 pairs)

Schema: exp_sel_data_out.json
  - datasets[].dataset: string (dataset name)
  - datasets[].examples[].input: string (JSON-encoded graph pair)
  - datasets[].examples[].output: string (ground truth label)
  - datasets[].examples[].metadata_*: optional per-example metadata
"""

import json
import resource
from pathlib import Path

# Resource limits (14GB RAM, 1 hour CPU)
resource.setrlimit(resource.RLIMIT_AS, (14 * 1024**3, 14 * 1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))

WORKSPACE = Path(__file__).parent
INPUT_FILE = WORKSPACE / "data_out.json"
OUTPUT_FILE = WORKSPACE / "full_data_out.json"


def build_input_string(pair: dict, graph_key: str) -> dict:
    """Build a compact but complete representation of a single graph for the input."""
    g = pair[graph_key]
    return {
        "name": g["name"],
        "num_nodes": g["num_nodes"],
        "num_edges": g["num_edges"],
        "degree_sequence": g["degree_sequence"],
        "edge_list": g["edge_list"],
        "adjacency_matrix": g["adjacency_matrix"],
        "adjacency_eigenvalues": g["adjacency_eigenvalues"],
        "laplacian_eigenvalues": g["laplacian_eigenvalues"],
    }


def pair_to_example(pair: dict, idx: int) -> dict:
    """Convert one graph pair to a schema-compliant example."""
    # Build input: JSON string of the graph pair
    input_data = {
        "pair_id": pair["pair_id"],
        "category": pair["category"],
        "task": "Determine if these two graphs are isomorphic.",
        "graph_a": build_input_string(pair, "graph_a"),
        "graph_b": build_input_string(pair, "graph_b"),
    }
    # Add skip_value for CSL graphs
    if "skip_value" in pair["graph_a"]:
        input_data["graph_a"]["skip_value"] = pair["graph_a"]["skip_value"]
    if "skip_value" in pair["graph_b"]:
        input_data["graph_b"]["skip_value"] = pair["graph_b"]["skip_value"]

    input_str = json.dumps(input_data, separators=(",", ":"))

    # Output: ground truth — these are all non-isomorphic pairs
    output_str = "non-isomorphic"

    # Metadata
    example = {
        "input": input_str,
        "output": output_str,
        "metadata_pair_id": pair["pair_id"],
        "metadata_category": pair["category"],
        "metadata_num_nodes": pair["graph_a"]["num_nodes"],
        "metadata_is_cospectral": pair["verification"]["is_cospectral_adjacency"],
        "metadata_wl1_equivalent": pair["verification"]["wl1_equivalent"],
        "metadata_row_index": idx,
        "metadata_task_type": "graph_isomorphism_testing",
        "metadata_graph_a_name": pair["graph_a"]["name"],
        "metadata_graph_b_name": pair["graph_b"]["name"],
    }

    # Add SRG or CSL specific metadata
    if "srg_parameters" in pair["verification"]:
        example["metadata_srg_parameters"] = pair["verification"]["srg_parameters"]
    if "csl_parameters" in pair["verification"]:
        example["metadata_csl_parameters"] = pair["verification"]["csl_parameters"]

    return example


def main() -> None:
    print(f"Loading data from: {INPUT_FILE}")
    raw = json.loads(INPUT_FILE.read_text())

    pairs = raw["pairs"]
    print(f"Total pairs loaded: {len(pairs)}")

    # Build all examples in a single dataset
    all_examples = []
    for idx, pair in enumerate(pairs):
        example = pair_to_example(pair, idx=idx)
        all_examples.append(example)

    print(f"  Total examples: {len(all_examples)}")

    # Assemble output — single dataset with all pairs
    output = {
        "metadata": {
            "description": (
                "1-WL Equivalent Graph Pairs Dataset: curated non-isomorphic graph pairs "
                "that are provably 1-WL equivalent, for testing expressiveness of graph "
                "distinguishing features beyond 1-WL. Spans 4 families: "
                "srg(16,6,2,2), srg(25,12,5,6), srg(26,10,3,4), and CSL(41,R)."
            ),
            "source": "Programmatically constructed from mathematical objects",
            "construction_sources": raw["metadata"]["construction_sources"],
            "references": raw["metadata"]["references"],
            "total_pairs": len(pairs),
            "categories": raw["metadata"]["categories"],
        },
        "datasets": [
            {
                "dataset": "1wl_equivalent_graph_pairs",
                "examples": all_examples,
            },
        ],
    }

    # Write output
    OUTPUT_FILE.write_text(json.dumps(output, indent=2))
    size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print(f"\nOutput saved: {OUTPUT_FILE} ({size_mb:.2f} MB)")
    print(f"  Dataset '1wl_equivalent_graph_pairs': {len(all_examples)} examples")

    # Quick validation
    loaded = json.loads(OUTPUT_FILE.read_text())
    assert "datasets" in loaded
    assert len(loaded["datasets"]) == 1
    ds = loaded["datasets"][0]
    assert ds["dataset"] == "1wl_equivalent_graph_pairs"
    assert "examples" in ds
    assert len(ds["examples"]) == 9
    for ex in ds["examples"]:
        assert "input" in ex
        assert "output" in ex
        assert isinstance(ex["input"], str)
        assert isinstance(ex["output"], str)
        # Verify input is valid JSON
        inp = json.loads(ex["input"])
        assert "graph_a" in inp
        assert "graph_b" in inp
    print("✓ Basic schema validation passed")


if __name__ == "__main__":
    main()
