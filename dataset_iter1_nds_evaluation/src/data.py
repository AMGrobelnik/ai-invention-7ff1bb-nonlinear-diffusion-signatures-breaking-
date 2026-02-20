# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Convert graph classification benchmark data (data_out.json) to the
exp_sel_data_out.json schema format.

Each graph becomes one example with:
  - input: JSON string of graph structure (num_nodes, edge_list, node_features)
  - output: class label as string
  - metadata_* fields: fold, feature_dim, num_classes, task_type, etc.
"""

import json
import resource
from pathlib import Path

# Resource limits: 14GB RAM, 1 hour CPU
resource.setrlimit(resource.RLIMIT_AS, (14 * 1024**3, 14 * 1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (3600, 3600))

WORKSPACE = Path("/home/adrian/projects/ai-inventor/aii_pipeline/runs/run__20260219_082247/3_invention_loop/iter_1/gen_art/data_id2_it1__opus")
INPUT_FILE = WORKSPACE / "data_out.json"
OUTPUT_FILE = WORKSPACE / "full_data_out.json"


def convert_graph_to_example(graph: dict, dataset_meta: dict) -> dict:
    """Convert a single graph dict to an example in the schema format."""
    # Build input: JSON string of the graph structure
    graph_input = {
        "num_nodes": graph["num_nodes"],
        "edge_list": graph["edge_list"],
        "node_features": graph["node_features"],
    }
    input_str = json.dumps(graph_input, separators=(",", ":"))

    # Build output: label as string
    output_str = str(graph["label"])

    # Build example with metadata
    example = {
        "input": input_str,
        "output": output_str,
        "metadata_fold": graph["fold"],
        "metadata_row_index": graph["id"],
        "metadata_task_type": "graph_classification",
        "metadata_n_classes": dataset_meta["num_classes"],
        "metadata_feature_dim": dataset_meta["feature_dim"],
        "metadata_has_node_features": dataset_meta["has_node_features"],
        "metadata_num_nodes": graph["num_nodes"],
        "metadata_num_edges": len(graph["edge_list"]),
        "metadata_fold_type": dataset_meta["fold_type"],
        "metadata_num_folds": dataset_meta["num_folds"],
    }

    return example


def convert_dataset(dataset_obj: dict) -> dict:
    """Convert one dataset object to the schema format."""
    dataset_name = dataset_obj["dataset_name"]
    print(f"  Converting {dataset_name}: {len(dataset_obj['graphs'])} graphs...")

    examples = []
    for graph in dataset_obj["graphs"]:
        example = convert_graph_to_example(graph, dataset_obj)
        examples.append(example)

    return {
        "dataset": dataset_name,
        "examples": examples,
    }


def main():
    print("=" * 60)
    print("Converting data_out.json → full_data_out.json")
    print("=" * 60)

    # Load input
    print(f"\nLoading: {INPUT_FILE}")
    with open(INPUT_FILE) as f:
        raw_data = json.load(f)

    print(f"Found {len(raw_data)} dataset objects")

    # Convert each dataset
    datasets = []
    total_examples = 0
    for ds_obj in raw_data:
        converted = convert_dataset(ds_obj)
        datasets.append(converted)
        total_examples += len(converted["examples"])

    # Build output object with optional top-level metadata
    output = {
        "metadata": {
            "description": "NDS Evaluation Graph Classification Benchmarks (CSL, MUTAG, PROTEINS, IMDB-BINARY) with full and mini versions",
            "source": "PyTorch Geometric (GNNBenchmarkDataset, TUDataset)",
            "num_datasets": len(datasets),
            "total_examples": total_examples,
            "random_seed": 42,
        },
        "datasets": datasets,
    }

    # Write output
    print(f"\nWriting: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, separators=(",", ":"))

    file_size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for ds in datasets:
        n_ex = len(ds["examples"])
        ex0 = ds["examples"][0]
        input_len = len(ex0["input"])
        print(f"  {ds['dataset']:20s} | {n_ex:5d} examples | input_len={input_len:6d} chars | output='{ex0['output']}'")
    print(f"\n  Total examples: {total_examples}")
    print("Done!")


if __name__ == "__main__":
    main()
