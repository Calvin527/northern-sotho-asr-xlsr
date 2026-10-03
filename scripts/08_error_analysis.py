from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from jiwer import process_words, wer

from common import ensure_columns, normalize_text


def _utterance_error_row(reference: str, prediction: str) -> dict:
    reference_norm = normalize_text(reference)
    prediction_norm = normalize_text(prediction)
    result = process_words(reference_norm, prediction_norm)
    substitutions = int(result.substitutions)
    deletions = int(result.deletions)
    insertions = int(result.insertions)
    hits = int(result.hits)
    reference_words = hits + substitutions + deletions
    total_errors = substitutions + deletions + insertions

    return {
        "reference_norm": reference_norm,
        "prediction_norm": prediction_norm,
        "reference_words": reference_words,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "total_word_errors": total_errors,
        "utterance_wer_percent": 100.0 * wer(reference_norm, prediction_norm),
        "exact_match": reference_norm == prediction_norm,
    }


def _select_examples(df: pd.DataFrame, count: int) -> pd.DataFrame:
    """Select deterministic, real examples spanning easy and difficult utterances."""
    if df.empty:
        return df

    candidates: list[pd.Series] = []
    exact = df.loc[df["exact_match"]]
    non_exact = df.loc[~df["exact_match"]].sort_values("utterance_wer_percent")

    if not exact.empty:
        candidates.append(exact.iloc[0])
    if not non_exact.empty:
        indices = sorted(
            set(
                int(round(x))
                for x in [
                    0,
                    (len(non_exact) - 1) * 0.25,
                    (len(non_exact) - 1) * 0.50,
                    (len(non_exact) - 1) * 0.75,
                    len(non_exact) - 1,
                ]
            )
        )
        for index in indices:
            candidates.append(non_exact.iloc[index])

    sample = pd.DataFrame(candidates).drop_duplicates(subset=["filename"], keep="first")
    return sample.head(count).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute word-level and utterance-level error analysis from test predictions."
    )
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--sample-count", type=int, default=6)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.predictions_csv)
    ensure_columns(df, ["filename", "reference", "prediction"])

    utterance_rows: list[dict] = []
    for _, row in df.iterrows():
        analysis = _utterance_error_row(row["reference"], row["prediction"])
        utterance_rows.append(
            {
                "filename": row["filename"],
                "reference": row["reference"],
                "prediction": row["prediction"],
                **analysis,
            }
        )

    utterance_df = pd.DataFrame(utterance_rows)

    substitutions = int(utterance_df["substitutions"].sum())
    deletions = int(utterance_df["deletions"].sum())
    insertions = int(utterance_df["insertions"].sum())
    total_errors = substitutions + deletions + insertions
    reference_words = int(utterance_df["reference_words"].sum())
    exact_matches = int(utterance_df["exact_match"].sum())

    def pct(value: int) -> float:
        return 0.0 if total_errors == 0 else 100.0 * value / total_errors

    summary = {
        "utterances": int(len(utterance_df)),
        "reference_word_tokens": reference_words,
        "total_word_errors": total_errors,
        "substitutions": substitutions,
        "substitution_percent_of_errors": pct(substitutions),
        "deletions": deletions,
        "deletion_percent_of_errors": pct(deletions),
        "insertions": insertions,
        "insertion_percent_of_errors": pct(insertions),
        "exact_matches": exact_matches,
        "exact_match_percent": 100.0 * exact_matches / len(utterance_df),
    }

    distribution = pd.DataFrame(
        [
            {"error_type": "Substitutions", "count": substitutions, "percentage_of_total_errors": pct(substitutions)},
            {"error_type": "Deletions", "count": deletions, "percentage_of_total_errors": pct(deletions)},
            {"error_type": "Insertions", "count": insertions, "percentage_of_total_errors": pct(insertions)},
            {"error_type": "Total Errors", "count": total_errors, "percentage_of_total_errors": 100.0 if total_errors else 0.0},
        ]
    )

    utterance_df.to_csv(output_dir / "utterance_error_analysis.csv", index=False)
    distribution.to_csv(output_dir / "word_error_analysis.csv", index=False)
    _select_examples(utterance_df, args.sample_count).to_csv(
        output_dir / "sample_transcript_comparison_table.csv",
        index=False,
    )
    (output_dir / "error_analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
