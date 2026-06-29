# Northern Sotho ASR using XLS-R 300M

This project develops and evaluates an end-to-end Automatic Speech Recognition (ASR) system for Northern Sotho/Sepedi using the pretrained `facebook/wav2vec2-xls-r-300m` model.

## Project Overview

The aim of this project is to fine-tune a multilingual pretrained speech model for Northern Sotho speech recognition. The system converts Northern Sotho speech audio into text transcripts using a CTC-based end-to-end ASR pipeline.

The project follows a full ASR workflow:

1. NCHLT Northern Sotho/Sepedi corpus preparation
2. XML-to-CSV conversion
3. Text cleaning and normalisation
4. Audio standardisation to 16 kHz
5. Character-level tokenizer development
6. XLS-R 300M fine-tuning using CTC loss
7. Evaluation using WER and CER
8. Transcript-level error analysis

## Dataset

The project uses the NCHLT Northern Sotho/Sepedi Speech Corpus, which contains speech audio recordings and corresponding orthographic transcriptions.

The raw dataset is not included in this repository because of file size and licensing considerations. Place the dataset in a Kaggle input path or local folder, then pass the path to the scripts.

Expected dataset structure:

```text
nchlt_nso/
├── audio/
│   ├── 001/
│   ├── 002/
│   └── ...
└── transcriptions/
    ├── nchlt_nso.trn.xml
    └── nchlt_nso.tst.xml
```

## Model

The model used in this project is:

```text
facebook/wav2vec2-xls-r-300m
```

The model was fine-tuned using CTC loss with a custom character-level tokenizer created from Northern Sotho transcriptions.

## Final Reported Results

The final model was trained for approximately 3 epochs, about 19,000 training steps.

| Metric | Result |
|---|---:|
| WER | 13.88% |
| CER | 3.74% |
| Test samples | 2,829 |
| Training samples | 50,655 |
| Validation samples | 5,629 |

## Repository Structure

```text
northern-sotho-asr-xlsr/
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── run_pipeline.ipynb
│
├── src/
│   ├── prepare_dataset.py
│   ├── train_xlsr.py
│   ├── evaluate_model.py
│   ├── error_analysis.py
│   └── create_chapter_assets.py
│
├── data/
│   ├── processed/
│   └── splits/
│
├── results/
│   ├── evaluation_summary_xlsr_nso_19000_steps.csv
│   └── error_summary_xlsr_nso_19000_steps.csv
│
├── chapter3_tables/
├── chapter3_figures/
├── chapter4_tables/
├── chapter4_figures/
├── chapter4_error_analysis/
└── docs/
    ├── chapter3_methodology_notes.md
    └── github_upload_guide.md
```

## Quick Start on Kaggle

Install dependencies:

```bash
pip install -r requirements.txt
```

Prepare the dataset:

```bash
python src/prepare_dataset.py \
  --dataset_dir /kaggle/input/asr-northern-sotho/nchlt_nso \
  --output_dir /kaggle/working/data
```

Fine-tune the model:

```bash
python src/train_xlsr.py \
  --data_dir /kaggle/working/data \
  --output_dir /kaggle/working/asr_model_xlsr_nso \
  --model_name facebook/wav2vec2-xls-r-300m \
  --max_steps 19000
```

Evaluate the model:

```bash
python src/evaluate_model.py \
  --model_dir /kaggle/working/asr_model_xlsr_nso/final \
  --test_csv /kaggle/working/data/splits/test.csv \
  --output_dir /kaggle/working/results
```

Run error analysis:

```bash
python src/error_analysis.py \
  --predictions_csv /kaggle/working/results/full_predictions_xlsr_nso_19000_steps.csv \
  --output_dir /kaggle/working/chapter4_error_analysis
```

## Notes

Large files are intentionally excluded from this repository, including the raw NCHLT audio files, Kaggle temporary working files, model checkpoints, and `model.safetensors` files.

If you want to share the trained model, use Kaggle outputs, Google Drive, or Hugging Face Model Hub instead of uploading the model directly to GitHub.
