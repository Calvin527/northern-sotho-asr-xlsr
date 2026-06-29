"""
Create report-ready Chapter 3 and Chapter 4 tables/figures.

This script is optional. It is useful for creating methodology and results screenshots
from CSV outputs.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_table_image(df: pd.DataFrame, title: str, output_path: Path, font_size: int = 8, scale_y: float = 1.5):
    fig_width = max(10, len(df.columns) * 2.6)
    fig_height = max(2.8, len(df) * 0.55 + 1.8)
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


def create_dataset_split_figure(output_dir: Path):
    splits = ["Training", "Validation", "Testing"]
    samples = [50655, 5629, 2829]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(splits, samples)
    ax.set_title("Dataset Split Distribution")
    ax.set_xlabel("Dataset Split")
    ax.set_ylabel("Number of Samples")
    ax.grid(axis="y")
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height, f"{int(height):,}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(output_dir / "fig_dataset_split_distribution.png", dpi=300)
    plt.close(fig)


def create_example_tables(chapter3_dir: Path, chapter4_dir: Path):
    table_3_1 = pd.DataFrame({
        "Research problem": [
            "Limited ASR tools for Northern Sotho / Sepedi",
            "Limited labelled speech data for low-resource languages",
            "Speech transcriptions stored in XML format",
            "Need for consistent text labels",
            "Need to adapt a model to Northern Sotho speech",
            "Need to measure transcription performance",
            "Need to understand model errors",
            "Need to test practical transcription speed",
        ],
        "Methodological response": [
            "Developed an end-to-end ASR system using XLS-R 300M",
            "Used the NCHLT Northern Sotho speech corpus",
            "Parsed XML transcription files and converted them to CSV",
            "Applied text cleaning and normalisation",
            "Fine-tuned facebook/wav2vec2-xls-r-300m using CTC loss",
            "Evaluated the model using WER and CER",
            "Analysed substitutions, deletions, insertions, and transcript examples",
            "Measured processing time and real-time factor",
        ],
        "Expected outcome": [
            "A working Northern Sotho speech-to-text model",
            "A usable supervised dataset for ASR training and testing",
            "Structured audio-text pairs for reproducible training",
            "Cleaner labels for tokenizer creation and model learning",
            "A model adapted to Northern Sotho speech patterns",
            "Quantitative evidence of model performance",
            "Clear understanding of common ASR mistakes",
            "Evidence of whether the model can transcribe efficiently",
        ],
    })
    table_3_1.to_csv(chapter3_dir / "table_3_1_problem_outcome_methodological_response.csv", index=False)
    save_table_image(table_3_1, "Table 3.1. Problem Outcome and Methodological Response", chapter3_dir / "table_3_1_problem_outcome_methodological_response.png", font_size=7, scale_y=2.2)

    summary = pd.DataFrame({
        "Metric": ["Model", "Training steps", "Approximate epochs", "Test samples", "WER", "CER"],
        "Value": ["facebook/wav2vec2-xls-r-300m", "19,000", "3.00", "2,829", "13.88%", "3.74%"],
    })
    summary.to_csv(chapter4_dir / "table_4_1_final_evaluation_summary.csv", index=False)
    save_table_image(summary, "Table 4.1. Final Evaluation Summary", chapter4_dir / "table_4_1_final_evaluation_summary.png", font_size=8)


def main():
    parser = argparse.ArgumentParser(description="Create example chapter assets.")
    parser.add_argument("--output_root", type=str, default=".")
    args = parser.parse_args()

    root = Path(args.output_root)
    ch3_tables = root / "chapter3_tables"
    ch3_figures = root / "chapter3_figures"
    ch4_tables = root / "chapter4_tables"
    ch3_tables.mkdir(parents=True, exist_ok=True)
    ch3_figures.mkdir(parents=True, exist_ok=True)
    ch4_tables.mkdir(parents=True, exist_ok=True)

    create_dataset_split_figure(ch3_figures)
    create_example_tables(ch3_tables, ch4_tables)
    print("Example chapter assets created.")


if __name__ == "__main__":
    main()
