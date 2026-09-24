# Northern Sotho XLS-R 300M ASR — Honours Research Reproducibility Package

This repository documents the final experimental configuration reported in:

**Fine-Tuning of an End-to-End Automatic Speech Recognition Model for Low-Resource Northern Sotho**

Author: **Maroka Musa Calvin**  
University of Limpopo — Honours of Science in Computer Science, 2026

## Important provenance note

This is a **reproducibility package reconstructed from the final research report and the verified final evaluation workflow**. It is not presented as an archival copy of the original training notebook/source tree.

The purpose of this repository is to make the final methodology, configuration, evaluation procedure, and reported results transparent and reproducible without misrepresenting an older experimental repository as the final accepted run.

The original NCHLT audio is **not redistributed** here.

## Research task

The project fine-tunes the multilingual self-supervised **facebook/wav2vec2-xls-r-300m** model for Northern Sotho (Sepedi) automatic speech recognition using the NCHLT Northern Sotho Speech Corpus.

The final evaluation also includes an external off-the-shelf **facebook/mms-1b-fl102** baseline using the released Northern Sotho (`nso`) adapter.

## Final documented dataset split

| Split | Utterances |
|---|---:|
| Training | 50,655 |
| Validation | 5,629 |
| Official test | 2,829 |
| Total | 59,113 |

The official test set contains **14,133 reference word tokens**.

## Final XLS-R 300M configuration

| Setting | Value |
|---|---|
| Base model | `facebook/wav2vec2-xls-r-300m` |
| Sampling rate | 16 kHz |
| Tokenizer | Character-level |
| Vocabulary size | 32 tokens |
| Word-boundary token | `|` |
| Special tokens | `[UNK]`, `[PAD]` |
| Loss | CTC |
| CTC reduction | Mean |
| CTC zero infinity | Enabled |
| Per-device batch size | 1 |
| Gradient accumulation | 4 |
| Effective batch size | 4 |
| Learning rate | `1e-4` |
| Scheduler | Linear |
| Warm-up ratio | 0.05 |
| Mixed precision | FP16 |
| Random seed | 42 |
| Feature extractor | Frozen |
| Evaluation/checkpoint interval | Every 900 steps |
| Maximum optimisation steps | 19,000 |
| Selected checkpoint | 8,100 steps |
| Decoding | Greedy CTC |
| External language model | None |
| Beam-search rescoring | None |
| Pronunciation lexicon | None |
| Fine-tuning hardware | NVIDIA RTX 2000 Ada Generation GPU |
| Fine-tuning environment | University JupyterLab |

## Final reported XLS-R results

| Metric | Result |
|---|---:|
| Validation WER | 21.69% |
| Validation CER | 5.60% |
| Test WER | 22.78% |
| Test CER | 5.86% |
| Exact-match utterances | 1,166 / 2,829 (41.22%) |
| Reference word tokens | 14,133 |
| Total word errors | 3,219 |
| Substitutions | 2,258 (70.15%) |
| Deletions | 852 (26.47%) |
| Insertions | 109 (3.39%) |

### XLS-R inference efficiency reported in the thesis

The RTF experiment used an **NVIDIA RTX 2000 Ada Generation GPU**, batch size 1, and timed only the model forward pass. Model loading and audio loading/preprocessing were excluded.

| Metric | Result |
|---|---:|
| Average audio duration | 3.716649 s |
| Average forward-pass time | 0.031816 s |
| Mean RTF | 0.008728 |
| Median RTF | 0.008631 |
| Minimum RTF | 0.006887 |
| Maximum RTF | 0.016939 |
| RTF < 1.0 | 100% |

## MMS-1B-FL102 baseline

The off-the-shelf `facebook/mms-1b-fl102` model was evaluated with the Northern Sotho `nso` adapter on the **same 2,829 official test utterances**.

No additional NCHLT fine-tuning was performed for MMS.

| Metric | Fine-tuned XLS-R 300M | MMS-1B-FL102 |
|---|---:|---:|
| WER | 22.78% | 46.85% |
| CER | 5.86% | 14.46% |
| Exact matches | 1,166 (41.22%) | 531 (18.77%) |
| Total word errors | 3,219 | 6,622 |
| Substitutions | 2,258 | 4,157 |
| Deletions | 852 | 1,185 |
| Insertions | 109 | 1,280 |

MMS baseline inference was executed on a **Tesla T4 in Kaggle**. Runtime is **not compared** with the XLS-R RTF results because the hardware environments and timing protocols differed. The cross-model comparison is restricted to transcription accuracy.

## Repository structure

```text
northern-sotho-xlsr-asr/
├── README.md
├── requirements.txt
├── .gitignore
├── config/
│   └── experiment.yaml
├── docs/
│   └── reproducibility_notes.md
├── manifests/
│   └── README.md
├── results/
│   └── final_metrics.json
└── scripts/
    ├── common.py
    ├── 01_prepare_nchlt.py
    ├── 02_build_tokenizer.py
    ├── 03_train_xlsr.py
    ├── 04_evaluate_xlsr.py
    ├── 05_evaluate_mms_baseline.py
    └── 06_make_comparison_figure.py
```

## Dataset layout

The scripts expect the NCHLT corpus to be available locally in a structure containing:

```text
nchlt_nso/
├── audio/
│   ├── <speaker_id>/
│   └── ...
└── transcriptions/
    ├── nchlt_nso.trn.xml
    └── nchlt_nso.tst.xml
```

The dataset itself is not included in this repository.

Dataset reference used in the report:

N.J. de Vries, M.H. Davel, J. Badenhorst, W.D. Basson, F. de Wet, E. Barnard and A. de Waal, “A smartphone-based ASR data collection tool for under-resourced languages,” *Speech Communication*, 56, 119–131, 2014. https://doi.org/10.1016/j.specom.2013.07.001

## Installation

Create a virtual environment, activate it, then install:

```bash
pip install -r requirements.txt
```

Exact package versions from the original training environment were not archived in the final report, so this repository does not invent version pins.

## Workflow

### 1. Prepare the NCHLT manifests

```bash
python scripts/01_prepare_nchlt.py \
  --dataset-root /path/to/dataset/root \
  --output-dir manifests \
  --make-internal-split
```

The official NCHLT test XML is preserved as the test set.

**Note:** the final report documents the train/validation counts but does not archive the exact membership of the original internal validation split. The optional deterministic split in this repository reconstructs the documented counts using seed 42; it must not be described as proof of the exact original row membership unless the original manifests are available.

### 2. Build the character tokenizer

```bash
python scripts/02_build_tokenizer.py \
  --train-csv manifests/train.csv \
  --validation-csv manifests/validation.csv \
  --output-dir artifacts/tokenizer
```

### 3. Fine-tune XLS-R 300M

```bash
python scripts/03_train_xlsr.py \
  --train-csv manifests/train.csv \
  --validation-csv manifests/validation.csv \
  --tokenizer-dir artifacts/tokenizer \
  --output-dir artifacts/xlsr-nso
```

### 4. Evaluate XLS-R

```bash
python scripts/04_evaluate_xlsr.py \
  --test-csv manifests/test.csv \
  --model-dir artifacts/xlsr-nso/checkpoint-8100 \
  --output-dir results/xlsr
```

### 5. Evaluate MMS baseline

```bash
python scripts/05_evaluate_mms_baseline.py \
  --test-csv manifests/test.csv \
  --output-dir results/mms
```

### 6. Create WER/CER comparison figure

```bash
python scripts/06_make_comparison_figure.py \
  --output results/xlsr_vs_mms_wer_cer.png
```

## Reproducibility boundaries

The following details are documented and encoded here: preprocessing rules, dataset counts, tokenizer design, core fine-tuning hyperparameters, random seed, decoding method, evaluation metrics, hardware used for final XLS-R training/inference, and the MMS baseline protocol.

The following details were **not recovered from the final archived report** and are therefore not fabricated here:

- exact wall-clock duration of the original XLS-R fine-tuning run;
- exact package-version freeze from the original training environment;
- exact row membership of the original internal train/validation split, unless original manifests are recovered.

See `docs/reproducibility_notes.md`.
