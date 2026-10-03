# Northern Sotho XLS-R 300M ASR — Honours Research Reproducibility Package

This repository documents the final experimental configuration reported in:

**Fine-Tuning of an End-to-End Automatic Speech Recognition Model for Low-Resource Northern Sotho**

**Author:** Maroka Musa Calvin  
**Institution:** University of Limpopo  
**Degree:** Honours of Science in Computer Science  
**Year:** 2026

---

## Repository scope and reproducibility note

This repository accompanies the honours research project **Fine-Tuning of an End-to-End Automatic Speech Recognition Model for Low-Resource Northern Sotho**.

It contains the project scripts, experiment configuration, tokenizer files, evaluation summaries, figures, and documentation used to describe and reproduce the reported Northern Sotho ASR workflow.

The original NCHLT speech corpus and trained model checkpoints are not redistributed in this repository.

A historical difference exists between the saved training notebook and the final submitted report for the internal train/validation allocation:

- saved notebook output: 50,656 training / 5,628 validation
- final submitted report: 50,655 training / 5,629 validation
- development total in both cases: 56,284
- official test set in both cases: 2,829

For this repository, the documented final report split of **50,655 training / 5,629 validation / 2,829 test utterances** is used as the reference configuration. The exact historical row membership of the original internal split is not claimed unless the original manifests are available.

---

## Research task

This project fine-tunes the multilingual self-supervised **`facebook/wav2vec2-xls-r-300m`** model for Northern Sotho (Sepedi) automatic speech recognition using the **NCHLT Northern Sotho Speech Corpus**.

The final evaluation also includes an external off-the-shelf **`facebook/mms-1b-fl102`** baseline using the released Northern Sotho **`nso`** adapter.

The main research objective is to investigate whether multilingual self-supervised speech representations can be adapted effectively to low-resource Northern Sotho ASR.

---

## Final documented dataset split

| Split | Utterances |
|---|---:|
| Training | 50,655 |
| Validation | 5,629 |
| Official test | 2,829 |
| **Total** | **59,113** |

The original development partition contains **56,284 utterances**, which was divided into training and internal validation subsets.

The official test set contains **2,829 utterances from 8 speakers** and **14,133 reference word tokens**.

In this project, an **utterance** refers to one speech-audio recording paired with its corresponding orthographic transcription.

---

## Text and audio preprocessing

The preprocessing pipeline follows the final research methodology:

- audio standardised to **16 kHz**
- transcripts converted to lowercase
- unnecessary punctuation removed
- repeated whitespace normalised
- empty transcripts removed
- Northern Sotho orthographic characters such as **`š`** preserved
- audio/transcript pairs verified before use
- official NCHLT test data kept separate from training and validation data

No NCHLT audio files are included in this repository.

---

## Character-level tokenizer

A character-level CTC tokenizer is used.

The final vocabulary contains **32 tokens**:

```text
|       -> 0
a       -> 1
b       -> 2
c       -> 3
d       -> 4
e       -> 5
f       -> 6
g       -> 7
h       -> 8
i       -> 9
j       -> 10
k       -> 11
l       -> 12
m       -> 13
n       -> 14
o       -> 15
p       -> 16
q       -> 17
r       -> 18
s       -> 19
t       -> 20
u       -> 21
v       -> 22
w       -> 23
x       -> 24
y       -> 25
z       -> 26
š       -> 27
[UNK]   -> 28
[PAD]   -> 29
<s>     -> 30
</s>    -> 31
```

The **`|`** token represents word boundaries.

During greedy CTC decoding:

1. the highest-probability token is selected at each acoustic time step;
2. repeated CTC labels are collapsed;
3. CTC blank predictions are removed;
4. `|` is converted back to ordinary whitespace;
5. the reconstructed text is normalised before WER and CER are calculated.

No fixed Northern Sotho word-level vocabulary, pronunciation lexicon, external language model, or beam-search rescoring is used.

---

## Final XLS-R 300M configuration

| Setting | Value |
|---|---|
| Base model | `facebook/wav2vec2-xls-r-300m` |
| Sampling rate | 16 kHz |
| Tokenizer | Character-level |
| Vocabulary size | 32 tokens |
| Word-boundary token | `|` |
| Special tokens | `[UNK]`, `[PAD]`, `<s>`, `</s>` |
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
| Gradient checkpointing | Enabled |
| Random seed | 42 |
| Convolutional feature encoder | Frozen |
| Transformer encoder | Fine-tuned |
| CTC output head | Fine-tuned |
| Evaluation/checkpoint interval | Every 900 steps |
| Maximum optimisation steps | 19,000 |
| Selected checkpoint | 8,100 steps |
| Decoding | Greedy CTC |
| External language model | None |
| Beam-search rescoring | None |
| Pronunciation lexicon | None |
| Fine-tuning hardware | NVIDIA RTX 2000 Ada Generation GPU |
| Fine-tuning environment | University JupyterLab |

---

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

A representative recognition error reported in the final analysis is:

```text
Reference:  programming language
Prediction: prokgremeng langwe
```

The much lower CER compared with WER indicates that many incorrect word predictions remain orthographically close to the reference transcription.

---

## XLS-R inference efficiency

The RTF experiment used an **NVIDIA RTX 2000 Ada Generation GPU**, inference batch size 1, and measured only the model forward-pass inference time.

Model loading and audio loading/preprocessing were excluded from the timing measurements.

| Metric | Result |
|---|---:|
| Mean audio duration | 3.716649 s |
| Mean forward-pass time | 0.031816 s |
| Mean RTF | 0.008728 |
| Median RTF | 0.008631 |
| Minimum RTF | 0.006887 |
| Maximum RTF | 0.016939 |
| RTF < 1.0 | 100% |

An RTF below 1.0 indicates faster-than-real-time processing under the evaluated hardware and timing protocol.

---

## MMS-1B-FL102 baseline

The off-the-shelf **`facebook/mms-1b-fl102`** model was evaluated with the Northern Sotho **`nso`** adapter on the **same 2,829 official test utterances**.

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

MMS baseline inference was executed on a **Tesla T4 GPU in Kaggle**.

Runtime is **not directly compared** with the XLS-R RTF results because the models were evaluated on different hardware and with different timing environments. The XLS-R versus MMS comparison is therefore restricted to **transcription accuracy**.

---

## Repository structure

```text
northern-sotho-asr-xlsr/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── configs/
│   └── training_config.json
│
├── manifests/
│   └── README.md
│
├── scripts/
│   ├── 01_prepare_dataset.py
│   ├── 02_create_split.py
│   ├── 03_preprocess_text.py
│   ├── 04_build_tokenizer.py
│   ├── 05_train_xlsr.py
│   ├── 06_evaluate_validation.py
│   ├── 07_evaluate_test.py
│   ├── 08_error_analysis.py
│   ├── 09_rtf_analysis.py
│   ├── 10_evaluate_mms_baseline.py
│   ├── 11_make_comparison_figure.py
│   └── common.py
│
├── tokenizer/
│   ├── vocab.json
│   ├── tokenizer_config.json
│   └── special_tokens_map.json
│
├── results/
│   ├── validation_metrics.csv
│   ├── test_metrics.csv
│   ├── rtf_summary.csv
│   ├── word_error_analysis.csv
│   ├── sample_transcript_comparison_table.csv
│   └── mms_baseline_results.csv
│
├── figures/
│   ├── figure_4_1_training_and_validation_loss_comparison.png
│   ├── figure_4_2_validation_wer_and_cer_comparison.png
│   ├── figure_4_3_validation_wer_convergence_and_best_checkpoint_selection.png
│   ├── figure_4_4_validation_and_official_test_error_rates.png
│   ├── figure_4_5_distribution_of_utterance_level_wer_on_official_test_set.png
│   ├── figure_4_6_official_test_recordings_by_utterance_error_category.png
│   ├── figure_4_7_distribution_of_word_level_errors_on_official_test_set.png
│   └── figure_4_8_xlsr_300m_vs_mms_1b_fl102_wer_cer_comparison.png
│
├── examples/
│   └── sample_predictions.csv
│
└── docs/
    ├── DATASET.md
    ├── EXPERIMENT.md
    └── REPRODUCIBILITY.md
```

Generated model checkpoints, downloaded model weights, raw NCHLT audio, temporary experiment files, and large training artifacts are intentionally excluded from Git tracking.

---

## Dataset layout

The scripts expect the NCHLT Northern Sotho corpus to be available locally.

A typical layout is:

```text
nchlt_nso/
├── audio/
│   ├── <speaker_id>/
│   └── ...
└── transcriptions/
    ├── nchlt_nso.trn.xml
    └── nchlt_nso.tst.xml
```

The exact local directory layout may vary depending on how the NCHLT archive was extracted. The dataset-preparation script should therefore receive the dataset root as a command-line argument rather than relying on a hard-coded university path.

The dataset itself is **not included** in this repository.

### Dataset reference

N. J. de Vries, M. H. Davel, J. Badenhorst, W. D. Basson, F. de Wet, E. Barnard, and A. de Waal, “A smartphone-based ASR data collection tool for under-resourced languages,” *Speech Communication*, vol. 56, pp. 119–131, 2014.

---

## Installation

Create and activate a Python virtual environment, then install the required packages:

```bash
pip install -r requirements.txt
```

Exact package versions from the original university training environment were not fully archived in the final report. The repository therefore avoids inventing an exact historical package freeze where it cannot be verified.

---

# Reproducibility workflow

## 1. Prepare the NCHLT dataset

Extract and verify valid audio/transcription pairs:

```bash
python scripts/01_prepare_dataset.py \
  --dataset-root /path/to/nchlt_nso \
  --output-dir manifests
```

This stage should:

- parse the NCHLT XML transcription files;
- verify referenced WAV files;
- remove missing/empty transcription entries;
- preserve the official test partition;
- create clean dataset records for later processing.

---

## 2. Create the internal train/validation split

```bash
python scripts/02_create_split.py \
  --input-csv manifests/development.csv \
  --output-dir manifests \
  --seed 42
```

The target split documented in the final report is:

```text
Training:   50,655
Validation:  5,629
```

The split implementation should keep speakers disjoint when reconstructing the report-aligned split.

**Important:** because the exact historical row membership of the submitted 50,655/5,629 partition was not archived, a reconstructed split must not be described as the exact original manifest.

---

## 3. Preprocess transcripts

```bash
python scripts/03_preprocess_text.py \
  --train-csv manifests/train.csv \
  --validation-csv manifests/validation.csv \
  --test-csv manifests/test.csv \
  --output-dir manifests/processed
```

This stage applies the text-normalisation rules used in the research:

- lowercase conversion;
- punctuation removal;
- whitespace normalisation;
- preservation of Northern Sotho characters such as `š`;
- removal of empty transcripts.

---

## 4. Build the tokenizer

```bash
python scripts/04_build_tokenizer.py \
  --train-csv manifests/processed/train.csv \
  --output-dir tokenizer
```

The tokenizer must reproduce the documented **32-token vocabulary**.

---

## 5. Fine-tune XLS-R 300M

```bash
python scripts/05_train_xlsr.py \
  --train-csv manifests/processed/train.csv \
  --validation-csv manifests/processed/validation.csv \
  --tokenizer-dir tokenizer \
  --config configs/training_config.json \
  --output-dir artifacts/xlsr-nso
```

The final documented run uses:

```text
Base model:                facebook/wav2vec2-xls-r-300m
Per-device batch size:     1
Gradient accumulation:     4
Effective batch size:      4
Learning rate:             1e-4
Warm-up ratio:             0.05
Maximum steps:             19,000
Evaluation interval:       900
FP16:                      enabled
Gradient checkpointing:    enabled
Random seed:               42
Feature encoder:           frozen
CTC reduction:             mean
CTC zero infinity:         enabled
```

The best reported validation checkpoint was **8,100 steps**.

---

## 6. Evaluate validation performance

```bash
python scripts/06_evaluate_validation.py \
  --validation-csv manifests/processed/validation.csv \
  --model-dir artifacts/xlsr-nso/checkpoint-8100 \
  --output results/validation_metrics.csv
```

Expected reported metrics for the selected checkpoint:

```text
Validation WER: 21.69%
Validation CER:  5.60%
```

---

## 7. Evaluate the official test set

```bash
python scripts/07_evaluate_test.py \
  --test-csv manifests/processed/test.csv \
  --model-dir artifacts/xlsr-nso/checkpoint-8100 \
  --output results/test_metrics.csv
```

Expected final reported metrics:

```text
Test WER: 22.78%
Test CER:  5.86%
```

The official test partition must remain untouched during training and model selection.

---

## 8. Run error analysis

```bash
python scripts/08_error_analysis.py \
  --predictions results/test_predictions.csv \
  --output-dir results
```

The final reported word-level error totals are:

```text
Reference words: 14,133
Total errors:     3,219
Substitutions:    2,258
Deletions:          852
Insertions:         109
Exact matches:    1,166 / 2,829
```

---

## 9. Run real-time factor analysis

```bash
python scripts/09_rtf_analysis.py \
  --test-csv manifests/processed/test.csv \
  --model-dir artifacts/xlsr-nso/checkpoint-8100 \
  --output results/rtf_summary.csv
```

The RTF implementation should time the **model forward pass only**.

Model loading and audio loading/preprocessing should remain outside the timed section so that the procedure is consistent with the reported experiment.

---

## 10. Evaluate the MMS baseline

```bash
python scripts/10_evaluate_mms_baseline.py \
  --test-csv manifests/processed/test.csv \
  --output results/mms_baseline_results.csv
```

This uses the released Northern Sotho **`nso`** adapter with the off-the-shelf MMS model and does **not** fine-tune MMS on NCHLT.

Expected reported transcription results:

```text
WER: 46.85%
CER: 14.46%
```

---

## 11. Create the XLS-R vs MMS comparison figure

```bash
python scripts/11_make_comparison_figure.py \
  --output figures/figure_4_8_xlsr_300m_vs_mms_1b_fl102_wer_cer_comparison.png
```

This comparison is for **WER and CER only**.

It should not imply a direct runtime comparison between XLS-R and MMS.

---

## Results and figures

The repository contains lightweight research outputs that can be distributed without redistributing the NCHLT corpus or model checkpoints.

Expected result files include:

```text
results/
├── validation_metrics.csv
├── test_metrics.csv
├── rtf_summary.csv
├── word_error_analysis.csv
├── sample_transcript_comparison_table.csv
└── mms_baseline_results.csv
```

The repository includes the following research figures:

```text
figures/
├── figure_4_1_training_and_validation_loss_comparison.png
├── figure_4_2_validation_wer_and_cer_comparison.png
├── figure_4_3_validation_wer_convergence_and_best_checkpoint_selection.png
├── figure_4_4_validation_and_official_test_error_rates.png
├── figure_4_5_distribution_of_utterance_level_wer_on_official_test_set.png
├── figure_4_6_official_test_recordings_by_utterance_error_category.png
├── figure_4_7_distribution_of_word_level_errors_on_official_test_set.png
└── figure_4_8_xlsr_300m_vs_mms_1b_fl102_wer_cer_comparison.png
```

Only genuine experiment outputs should be committed. Missing per-utterance results should **not** be fabricated simply to populate a file.

---

## Files intentionally excluded from GitHub

The repository should not contain:

```text
*.wav
*.zip
checkpoint-*/
*.pt
*.bin
*.safetensors
wandb/
__pycache__/
.ipynb_checkpoints/
*.ipynb
```

It should also avoid committing:

- the NCHLT speech corpus;
- model checkpoints;
- downloaded Hugging Face model weights;
- private university filesystem paths;
- large generated training artifacts;
- temporary debugging outputs;
- the raw historical training notebook.

---

## Reproducibility boundaries

The following aspects are documented and represented in the clean repository:

- NCHLT Northern Sotho data preparation workflow;
- final documented dataset counts;
- 16 kHz audio processing;
- Northern Sotho text normalisation;
- 32-token character-level tokenizer;
- XLS-R 300M model configuration;
- CTC training objective;
- CTC mean reduction and zero-infinity setting;
- physical and effective batch size;
- learning rate and scheduler;
- warm-up ratio;
- FP16;
- gradient checkpointing;
- random seed;
- frozen convolutional feature encoder;
- validation/checkpoint interval;
- maximum optimisation steps;
- selected checkpoint;
- greedy CTC decoding;
- WER and CER evaluation;
- word-level error analysis;
- RTF methodology;
- MMS-1B-FL102 baseline protocol.

The following details should **not** be fabricated:

- exact wall-clock duration of the original XLS-R fine-tuning run;
- exact package-version freeze if not recoverable from the original environment;
- exact historical row membership of the final submitted train/validation split;
- any missing per-utterance prediction or timing records that were not preserved.

See **`docs/REPRODUCIBILITY.md`** for additional details.

---

## Citation

If this repository is used in academic work, cite the corresponding Northern Sotho ASR research and the NCHLT corpus source.

### NCHLT corpus

N. J. de Vries, M. H. Davel, J. Badenhorst, W. D. Basson, F. de Wet, E. Barnard, and A. de Waal, “A smartphone-based ASR data collection tool for under-resourced languages,” *Speech Communication*, vol. 56, pp. 119–131, 2014.

### XLS-R

A. Babu *et al.*, “XLS-R: Self-supervised cross-lingual speech representation learning at scale,” in *Proceedings of Interspeech*, 2022.

### MMS

V. Pratap *et al.*, “Scaling Speech Technology to 1,000+ Languages,” *Journal of Machine Learning Research*, vol. 25, no. 97, pp. 1–52, 2024.

---

## Author

**Maroka Musa Calvin**  
Department of Computer Science  
University of Limpopo  
2026
