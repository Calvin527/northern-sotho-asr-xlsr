# Reproducibility Notes

This package is intentionally transparent about what is documented and what is not.

## Recovered/documented final experiment details

The final report documents the NCHLT dataset, split counts, 16 kHz audio, a 32-token character tokenizer, XLS-R 300M, CTC fine-tuning, batch size 1, gradient accumulation 4, learning rate 1e-4, warm-up ratio 0.05, linear decay, FP16, seed 42, frozen convolutional feature extractor, mean CTC reduction, zero-infinity masking, 900-step validation/checkpoint intervals, 19,000 maximum steps, best checkpoint at 8,100 steps, greedy CTC decoding, no external language model, no beam-search rescoring, no pronunciation lexicon, NVIDIA RTX 2000 Ada Generation hardware for the final XLS-R experiment, and the MMS-1B-FL102 Northern Sotho baseline.

## Not recovered from the final archive

The following are deliberately not invented:

1. Original training wall-clock duration.
2. Exact package-version freeze from the original university JupyterLab run.
3. Exact row membership of the original internal validation subset, unless the original manifests are later recovered.
4. An archival source-code snapshot of the final accepted training run.

For this reason, this repository is best described as a **reproducibility package documenting the final experimental configuration**, not as an untouched archival snapshot of the original experiment.

## Internal split reconstruction

`01_prepare_nchlt.py --make-internal-split` creates a deterministic utterance-level split with the documented sizes and seed 42.

This is useful for reproducing the methodology, but it should not be claimed to reproduce the exact historical validation membership unless the original manifests are recovered.

## Runtime fairness

The thesis reports XLS-R RTF using the NVIDIA RTX 2000 Ada Generation GPU and a forward-pass-only timing protocol.

The MMS baseline was executed on a Tesla T4 in Kaggle using a different runtime context.

Therefore this repository does not make a direct XLS-R-vs-MMS speed claim. The cross-model baseline comparison is restricted to transcription accuracy.
