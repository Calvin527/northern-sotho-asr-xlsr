# Chapter 3 Methodology Notes

## Key Project Details

- Dataset: NCHLT Northern Sotho/Sepedi Speech Corpus
- Total usable audio-text pairs: 59,113
- Training samples: 50,655
- Validation samples: 5,629
- Test samples: 2,829
- Model: `facebook/wav2vec2-xls-r-300m`
- Short name: XLS-R 300M
- Framework: PyTorch + Hugging Face Transformers
- Training objective: CTC loss
- Final training: 19,000 steps / approximately 3 epochs
- Evaluation: WER, CER, substitution/deletion/insertion analysis, transcript comparison, Real-Time Factor

## Recommended Chapter 3 Headings

1. Introduction
2. Research Design
3. Research Outcome and System Objective
4. Dataset Requirement
5. Dataset Selection and Justification
6. Data Extraction Process
7. XML-to-CSV Conversion
8. Text Cleaning and Normalisation
9. Audio Processing and Standardisation
10. Dataset Segmentation
11. Speech Signal Analysis
12. Tokenizer Development
13. Model Selection
14. System Development Process
15. Fine-Tuning Configuration
16. Model Saving and Reproducibility
17. Evaluation Method
18. Analysis Techniques and Justification
19. Software and Hardware Environment
20. Chapter Summary

## Figure and Table Labels

- Figure 3.1: End-to-end ASR pipeline for Northern Sotho speech recognition.
- Figure 3.2: Research methodology pipeline for the Northern Sotho ASR system.
- Figure 3.3: Dataset split distribution for training, validation, and testing.
- Figure 3.4: Waveform representation of a Northern Sotho speech sample.
- Figure 3.5: Spectrogram representation of a Northern Sotho speech sample.
- Figure 3.6: Character-level tokenizer vocabulary developed from Northern Sotho transcriptions.
- Figure 3.7: System development process for the Northern Sotho ASR system.

- Table 3.1: Problem outcome and methodological response.
- Table 3.2: Summary of the NCHLT Northern Sotho/Sepedi Speech Corpus.
- Table 3.3: Dataset split used for training, validation, and testing.
- Table 3.4: Sample CSV file after XML-to-CSV conversion.
- Table 3.5: Sample amplitude values extracted from a Northern Sotho speech file.
- Table 3.6: Amplitude summary statistics of the selected speech signal.
- Table 3.7: Selected pretrained XLS-R 300M model specification.
- Table 3.8: Fine-tuning configuration for the Northern Sotho XLS-R model.
- Table 3.9: Evaluation metrics used to analyse model performance.
- Table 3.10: Software and hardware environment used in the study.
