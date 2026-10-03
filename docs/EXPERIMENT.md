# Experiment

## Model

The main model is **`facebook/wav2vec2-xls-r-300m`**.

The model is fine-tuned for Northern Sotho automatic speech recognition using Connectionist Temporal Classification (CTC).

## Tokenizer

A **32-token character-level tokenizer** is used.

The vocabulary contains:

- `|` as the word-boundary token;
- `a-z`;
- `š`;
- `[UNK]`;
- `[PAD]`;
- `<s>`;
- `</s>`.

During greedy CTC decoding, repeated labels and blank predictions are collapsed, and `|` is converted back to whitespace.

No external language model, pronunciation lexicon, or beam-search rescoring is used.

## Training configuration

| Setting | Value |
|---|---|
| Base model | `facebook/wav2vec2-xls-r-300m` |
| Sampling rate | 16 kHz |
| Physical batch size | 1 |
| Gradient accumulation | 4 |
| Effective batch size | 4 |
| Learning rate | `1e-4` |
| Learning-rate scheduler | Linear |
| Warm-up ratio | 0.05 |
| Maximum optimisation steps | 19,000 |
| Evaluation/checkpoint interval | 900 steps |
| Mixed precision | FP16 |
| Gradient checkpointing | Enabled |
| Random seed | 42 |
| CTC loss reduction | Mean |
| CTC zero infinity | Enabled |
| Convolutional feature encoder | Frozen |
| Transformer encoder | Fine-tuned |
| CTC output head | Fine-tuned |
| Selected checkpoint | 8,100 steps |
| Training hardware | NVIDIA RTX 2000 Ada Generation GPU |
| Training environment | University JupyterLab |

## Validation and test performance

| Metric | Validation | Official test |
|---|---:|---:|
| WER | 21.69% | 22.78% |
| CER | 5.60% | 5.86% |

The selected checkpoint at **8,100 steps** produced the lowest reported validation WER.

## Official-test error analysis

The official test set contains **2,829 utterances** and **14,133 reference word tokens**.

| Error type | Count | Percentage |
|---|---:|---:|
| Substitutions | 2,258 | 70.15% |
| Deletions | 852 | 26.47% |
| Insertions | 109 | 3.39% |
| **Total** | **3,219** | **100%** |

The system produced **1,166 exact-match utterances (41.22%)**.

## Real-time factor evaluation

XLS-R inference timing was measured on an **NVIDIA RTX 2000 Ada Generation GPU** with batch size 1.

The timing protocol includes only the model forward pass. Model loading and audio loading/preprocessing are excluded.

| Measure | Result |
|---|---:|
| Mean audio duration | 3.716649 s |
| Mean forward-pass time | 0.031816 s |
| Mean RTF | 0.008728 |
| Median RTF | 0.008631 |
| Minimum RTF | 0.006887 |
| Maximum RTF | 0.016939 |
| RTF < 1.0 | 100% |

## MMS baseline

The off-the-shelf **`facebook/mms-1b-fl102`** model is evaluated with the released Northern Sotho `nso` adapter.

No NCHLT fine-tuning is performed for MMS.

| Metric | XLS-R 300M | MMS-1B-FL102 |
|---|---:|---:|
| WER | 22.78% | 46.85% |
| CER | 5.86% | 14.46% |
| Exact matches | 1,166 (41.22%) | 531 (18.77%) |
| Total word errors | 3,219 | 6,622 |

The MMS baseline was run on a **Tesla T4 GPU in Kaggle**. Runtime is not directly compared with the XLS-R RTF experiment because the hardware and timing environments differ.
