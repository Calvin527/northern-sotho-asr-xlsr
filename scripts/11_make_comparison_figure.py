from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("results/xlsr_vs_mms_wer_cer.png"))
    args = p.parse_args(); args.output.parent.mkdir(parents=True, exist_ok=True)
    models = ["Fine-tuned XLS-R 300M", "MMS-1B-FL102"]
    wer_values = [22.78, 46.85]
    cer_values = [5.86, 14.46]
    x = np.arange(len(models)); width = 0.32
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    wer_bars = ax.bar(x - width/2, wer_values, width, label="WER (%)")
    cer_bars = ax.bar(x + width/2, cer_values, width, label="CER (%)")
    ax.set_xlabel("ASR Model", fontsize=10); ax.set_ylabel("Error Rate (%)", fontsize=10)
    ax.set_title("WER and CER Comparison", fontsize=11)
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=9); ax.set_ylim(0, 55)
    ax.grid(axis="y", linestyle="--", alpha=0.4); ax.legend(fontsize=9)
    for bars in (wer_bars, cer_bars):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h+0.6, f"{h:.2f}%", ha="center", va="bottom", fontsize=9)
    plt.tight_layout(); plt.savefig(args.output, dpi=300, bbox_inches="tight"); plt.show()
    print("Saved:", args.output.resolve())


if __name__ == "__main__":
    main()
