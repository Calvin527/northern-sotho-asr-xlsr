"""
Create transcript-level error analysis for Northern Sotho ASR predictions.

The script identifies correct words, substitutions, deletions, insertions,
and produces screenshot-ready tables and a bar chart.
"""

import argparse
from difflib import SequenceMatcher
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def analyse_sentence_errors(reference: str, prediction: str) -> dict:
    ref_words = str(reference).split()
    pred_words = str(prediction).split()
    matcher = SequenceMatcher(None, ref_words, pred_words)

    correct_words = []
    substitutions = []
    deletions = []
    insertions = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            correct_words.extend(ref_words[i1:i2])
        elif tag == "replace":
            ref_part = ref_words[i1:i2]
            pred_part = pred_words[j1:j2]
            max_len = max(len(ref_part), len(pred_part))
            for k in range(max_len):
                ref_word = ref_part[k] if k < len(ref_part) else ""
                pred_word = pred_part[k] if k < len(pred_part) else ""
                if ref_word and pred_word:
                    substitutions.append(f"{ref_word} → {pred_word}")
                elif ref_word:
                    deletions.append(ref_word)
                elif pred_word:
                    insertions.append(pred_word)
        elif tag == "delete":
            deletions.extend(ref_words[i1:i2])
        elif tag == "insert":
            insertions.extend(pred_words[j1:j2])

    return {
        "correct_words": ", ".join(correct_words) if correct_words else "None",
        "substitutions": ", ".join(substitutions) if substitutions else "None",
        "deletions": ", ".join(deletions) if deletions else "None",
        "insertions": ", ".join(insertions) if insertions else "None",
        "suggested_correction": reference,
    }


def save_table_image(df: pd.DataFrame, title: str, output_path: Path, font_size: int = 7, scale_y: float = 1.8):
    fig_width = max(12, len(df.columns) * 2.4)
    fig_height = max(4, len(df) * 0.75 + 2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc="left", colLoc="left", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, scale_y)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_text_props(weight="bold", color="black")
            cell.set_facecolor("#4C9BD6")

    plt.title(title, fontsize=13, fontstyle="italic", loc="left", pad=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create ASR transcript error analysis.")
    parser.add_argument("--predictions_csv", type=str, required=True, help="CSV with reference and prediction columns.")
    parser.add_argument("--output_dir", type=str, default="chapter4_error_analysis")
    args = parser.parse_args()

    predictions_csv = Path(args.predictions_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not predictions_csv.exists():
        raise FileNotFoundError(f"Prediction file not found: {predictions_csv}")

    df = pd.read_csv(predictions_csv)
    for col in ["reference", "prediction"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df["reference"] = df["reference"].fillna("").astype(str).str.lower().str.strip()
    df["prediction"] = df["prediction"].fillna("").astype(str).str.lower().str.strip()

    rows = []
    for _, row in df.iterrows():
        result = analyse_sentence_errors(row["reference"], row["prediction"])
        rows.append({
            "audio_path": row.get("audio_path", ""),
            "reference": row["reference"],
            "prediction": row["prediction"],
            "sentence_wer": row.get("sentence_wer", np.nan),
            "sentence_cer": row.get("sentence_cer", np.nan),
            **result,
        })

    analysis_df = pd.DataFrame(rows)
    analysis_df.to_csv(output_dir / "full_word_level_error_analysis.csv", index=False)

    correct_examples = analysis_df[analysis_df["reference"] == analysis_df["prediction"]].head(5)
    minor_errors = analysis_df[
        (analysis_df["reference"] != analysis_df["prediction"])
        & (analysis_df["sentence_wer"].fillna(1) > 0)
        & (analysis_df["sentence_wer"].fillna(1) <= 0.5)
    ].head(5)
    major_errors = analysis_df[
        (analysis_df["reference"] != analysis_df["prediction"])
        & (analysis_df["sentence_wer"].fillna(0) > 0.5)
    ].head(5)

    report_examples = pd.concat([correct_examples, minor_errors, major_errors], ignore_index=True)
    report_examples.to_csv(output_dir / "error_analysis_examples_for_report.csv", index=False)

    total = len(analysis_df)
    correct = int((analysis_df["reference"] == analysis_df["prediction"]).sum())
    with_errors = total - correct
    with_sub = int((analysis_df["substitutions"] != "None").sum())
    with_del = int((analysis_df["deletions"] != "None").sum())
    with_ins = int((analysis_df["insertions"] != "None").sum())

    summary_df = pd.DataFrame({
        "Category": [
            "Total test samples",
            "Fully correct transcripts",
            "Transcripts with errors",
            "Samples with substitutions",
            "Samples with deletions",
            "Samples with insertions",
        ],
        "Count": [total, correct, with_errors, with_sub, with_del, with_ins],
        "Percentage": [
            "100%",
            f"{(correct / total) * 100:.2f}%",
            f"{(with_errors / total) * 100:.2f}%",
            f"{(with_sub / total) * 100:.2f}%",
            f"{(with_del / total) * 100:.2f}%",
            f"{(with_ins / total) * 100:.2f}%",
        ],
    })
    summary_df.to_csv(output_dir / "error_analysis_summary.csv", index=False)

    save_table_image(summary_df, "Table 4.X. Error Analysis Summary", output_dir / "table_error_analysis_summary.png", font_size=8)

    examples_for_image = report_examples[[
        "reference", "prediction", "correct_words", "substitutions", "deletions", "insertions", "suggested_correction"
    ]].copy()
    save_table_image(examples_for_image, "Table 4.X. Transcript Error Analysis Examples", output_dir / "table_transcript_error_analysis_examples.png", font_size=6, scale_y=2.2)

    plot_df = pd.DataFrame({
        "Error type": ["Substitution", "Deletion", "Insertion"],
        "Number of samples": [with_sub, with_del, with_ins],
    })
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(plot_df["Error type"], plot_df["Number of samples"])
    ax.set_title("Error Type Occurrence in Test Samples")
    ax.set_xlabel("Error Type")
    ax.set_ylabel("Number of Samples")
    ax.grid(axis="y")
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, str(int(height)), ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(output_dir / "fig_error_type_occurrence.png", dpi=300)
    plt.close(fig)

    print("Error analysis complete. Outputs saved to:", output_dir)


if __name__ == "__main__":
    main()
