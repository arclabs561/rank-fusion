#!/usr/bin/env python3
"""
Convert HuggingFace dataset to TREC format.

Usage:
    python convert_hf_to_trec.py --dataset mteb/miracl --split dev --language en --output-dir ./datasets/miracl-en
"""

import argparse
import json
from pathlib import Path
from datasets import load_dataset


def convert_to_trec_runs(dataset, output_path: Path, run_tag: str = "hf_run"):
    """
    Convert HuggingFace dataset to TREC runs format.
    
    NOTE: This creates runs from positive passages, which is useful for qrels
    but NOT for actual retrieval evaluation. For real evaluation, you need to:
    1. Run a retrieval system (BM25, dense retrieval, etc.) on the corpus
    2. Generate actual retrieval runs with scores
    3. Then use those runs for fusion evaluation
    
    This function is primarily for creating qrels from positive passages.
    """
    runs_file = output_path / "runs.txt"
    
    # Group by query_id first
    by_query = {}
    for example in dataset:
        query_id = str(example.get("query_id", example.get("_id", "")))
        if query_id not in by_query:
            by_query[query_id] = []
        
        # Handle different dataset structures
        if "positive_passages" in example:
            # MIRACL/LoTTE style - these are actually qrels, not runs
            # We'll create synthetic runs for demonstration
            for rank, passage in enumerate(example["positive_passages"][:100]):
                doc_id = passage.get("docid", passage.get("doc_id", ""))
                score = 1.0 - (rank * 0.01)  # Simple scoring
                by_query[query_id].append((doc_id, score, rank))
        elif "corpus_id" in example:
            # BEIR style - actual retrieval result
            doc_id = example["corpus_id"]
            score = example.get("score", 1.0)
            rank = example.get("rank", 1)
            by_query[query_id].append((doc_id, score, rank))
    
    # Write grouped and sorted by query
    with open(runs_file, "w") as f:
        for query_id in sorted(by_query.keys()):
            # Sort by score descending within each query
            entries = by_query[query_id]
            entries.sort(key=lambda x: x[1], reverse=True)
            
            for rank, (doc_id, score, _) in enumerate(entries, start=1):
                f.write(f"{query_id} Q0 {doc_id} {rank} {score:.6f} {run_tag}\n")


def convert_to_trec_qrels(dataset, output_path: Path):
    """Convert HuggingFace dataset to TREC qrels format."""
    qrels_file = output_path / "qrels.txt"
    
    with open(qrels_file, "w") as f:
        for example in dataset:
            query_id = str(example.get("query_id", example.get("_id", "")))
            
            # Handle positive passages
            if "positive_passages" in example:
                for passage in example["positive_passages"]:
                    doc_id = passage.get("docid", passage.get("doc_id", ""))
                    relevance = passage.get("score", 1)  # Default to relevant
                    f.write(f"{query_id} 0 {doc_id} {relevance}\n")
            elif "positive" in example:
                doc_id = example["positive"].get("docid", example["positive"].get("_id", ""))
                relevance = example.get("score", 1)
                f.write(f"{query_id} 0 {doc_id} {relevance}\n")


def main():
    parser = argparse.ArgumentParser(description="Convert HuggingFace dataset to TREC format")
    parser.add_argument("--dataset", required=True, help="HuggingFace dataset ID (e.g., mteb/miracl)")
    parser.add_argument("--split", default="dev", help="Dataset split (dev, test, train)")
    parser.add_argument("--language", help="Language code (for multilingual datasets)")
    parser.add_argument("--output-dir", required=True, help="Output directory for TREC files")
    parser.add_argument("--run-tag", default="hf_run", help="Run tag for TREC runs file")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading dataset: {args.dataset}")
    
    # Load dataset
    if args.language:
        dataset = load_dataset(args.dataset, args.language, split=args.split)
    else:
        dataset = load_dataset(args.dataset, split=args.split)
    
    print(f"Dataset loaded: {len(dataset)} examples")
    
    # Convert to TREC format
    print("Converting to TREC runs format...")
    convert_to_trec_runs(dataset, output_dir, args.run_tag)
    
    print("Converting to TREC qrels format...")
    convert_to_trec_qrels(dataset, output_dir)
    
    print(f"\nConversion complete!")
    print(f"  Runs: {output_dir / 'runs.txt'}")
    print(f"  Qrels: {output_dir / 'qrels.txt'}")
    print(f"\nNext step: Run evaluation with:")
    print(f"  cargo run --bin evaluate-real-world -- --datasets-dir {output_dir.parent}")


if __name__ == "__main__":
    main()

