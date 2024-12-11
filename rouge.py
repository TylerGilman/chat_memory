import json
from rouge_score import rouge_scorer
import pandas as pd
import numpy as np
from typing import Dict, List


def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from JSONL file."""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def calculate_rouge_scores(generated_summary: str, reference_summary: str) -> Dict:
    """Calculate multiple ROUGE scores between generated and reference summaries."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference_summary, generated_summary)

    return {
        "rouge1_precision": scores["rouge1"].precision,
        "rouge1_recall": scores["rouge1"].recall,
        "rouge1_fmeasure": scores["rouge1"].fmeasure,
        "rouge2_precision": scores["rouge2"].precision,
        "rouge2_recall": scores["rouge2"].recall,
        "rouge2_fmeasure": scores["rouge2"].fmeasure,
        "rougeL_precision": scores["rougeL"].precision,
        "rougeL_recall": scores["rougeL"].recall,
        "rougeL_fmeasure": scores["rougeL"].fmeasure,
    }


def analyze_summaries(file_path: str) -> pd.DataFrame:
    """Analyze summaries and calculate statistics."""
    data = load_jsonl(file_path)
    results = []

    for item in data:
        generated_summary = item["generated_summary"]
        original_summary = item["original_summary"]

        scores = calculate_rouge_scores(generated_summary, original_summary)
        scores["id"] = item["id"]

        results.append(scores)

    return pd.DataFrame(results)


def print_statistics(df: pd.DataFrame):
    """Print detailed statistics of ROUGE scores."""
    metrics = ["rouge1", "rouge2", "rougeL"]
    aspects = ["precision", "recall", "fmeasure"]

    print("Summary Statistics:")
    print("=" * 80)

    for metric in metrics:
        print(f"\n{metric.upper()} Scores:")
        print("-" * 40)

        for aspect in aspects:
            col = f"{metric}_{aspect}"
            stats = df[col].describe()

            print(f"\n{aspect.capitalize()}:")
            print(f"  Mean: {stats['mean']:.4f}")
            print(f"  Median: {stats['50%']:.4f}")
            print(f"  Std Dev: {stats['std']:.4f}")
            print(f"  Min: {stats['min']:.4f}")
            print(f"  Max: {stats['max']:.4f}")


def main():
    file_path = "results/test_truth.jsonl"

    # Analyze summaries
    results_df = analyze_summaries(file_path)

    # Print statistics
    print_statistics(results_df)

    # Save detailed results
    results_df.to_csv("rouge_scores_analysis.csv", index=False)
    print("\nDetailed results saved to rouge_scores_analysis.csv")


if __name__ == "__main__":
    main()
