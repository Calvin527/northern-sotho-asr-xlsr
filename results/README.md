# Results folder

This folder contains lightweight **reported final-study outputs** that can be
shared without redistributing the NCHLT corpus or model checkpoints.

The CSV files contain aggregate metrics and representative transcript examples
that are explicitly documented in the final honours report / camera-ready paper.

Important boundaries:

- `validation_metrics.csv` and `test_metrics.csv` record the reported final
  validation/test metrics for checkpoint 8100.
- `rtf_summary.csv` contains only the reported aggregate RTF statistics. It does
  not fabricate missing per-utterance timing rows.
- `word_error_analysis.csv` contains the reported corpus-level substitution,
  deletion, and insertion totals.
- `sample_transcript_comparison_table.csv` contains genuine representative
  transcript pairs recovered from Table 4.6.
- `mms_baseline_results.csv` contains the reported off-the-shelf
  MMS-1B-FL102 `nso` baseline results.
- Runtime must not be directly compared between XLS-R and MMS because they were
  evaluated on different hardware / timing environments.

Generated per-utterance predictions or timings should only be added when they
come from preserved or newly executed evaluation outputs.
