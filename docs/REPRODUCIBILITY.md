# Reproducibility

## Scope

This repository documents the final Northern Sotho XLS-R 300M research workflow, configuration, tokenizer, evaluation procedures, reported metrics, figures, and representative transcript examples.

The repository is intended to support transparent documentation and reproduction of the reported methodology.

## What is included

The repository documents:

- NCHLT Northern Sotho data preparation;
- the final reported dataset counts;
- 16 kHz audio processing;
- transcript normalisation;
- preservation of Northern Sotho characters such as `š`;
- the 32-token character-level tokenizer;
- XLS-R 300M fine-tuning settings;
- CTC configuration;
- batch size and gradient accumulation;
- learning-rate schedule and warm-up ratio;
- FP16 and gradient checkpointing;
- random seed 42;
- the frozen convolutional feature encoder;
- evaluation/checkpoint intervals;
- the selected checkpoint;
- greedy CTC decoding;
- validation and official-test WER/CER;
- word-level error analysis;
- real-time-factor methodology;
- MMS-1B-FL102 baseline evaluation;
- representative transcript comparisons.

## What is not redistributed

The repository does not include:

- NCHLT WAV files;
- the NCHLT speech corpus archive;
- XLS-R model checkpoints;
- downloaded model weights;
- MMS model weights;
- private university filesystem paths;
- large temporary training artifacts;
- the historical raw notebook.

## Historical train/validation split difference

The saved training notebook records:

```text
Training:   50,656
Validation:  5,628
Total:      56,284
```

The final submitted report records:

```text
Training:   50,655
Validation:  5,629
Total:      56,284
```

The official test set contains **2,829 utterances** in both cases.

This is a one-utterance difference in the internal development split. The final submitted report values are used as the reference configuration in this repository.

The exact historical row membership of the final submitted internal split is not claimed unless the original manifests are recovered.

## Results policy

Only genuine reported or preserved outputs should be committed.

Aggregate metrics documented in the final study can be stored in CSV or JSON form.

Missing per-utterance predictions, timings, or manifests should not be invented to make the repository appear more complete.

Newly generated outputs from reproduction runs should be clearly identified as reproduction outputs rather than historical results.

## Environment limitations

The exact historical package freeze and exact wall-clock training duration were not fully preserved in the final report.

The repository therefore does not invent package-version pins or training-duration values that cannot be verified.

## Runtime comparison boundary

The XLS-R RTF evaluation used an NVIDIA RTX 2000 Ada Generation GPU.

The MMS baseline was evaluated on a Tesla T4 in Kaggle.

Because the hardware and timing protocols differ, the XLS-R versus MMS comparison is limited to transcription accuracy and should not be presented as a direct runtime comparison.
