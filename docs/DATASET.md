# Dataset

## NCHLT Northern Sotho Speech Corpus

This project uses the **NCHLT Northern Sotho / Sepedi Speech Corpus**.

The corpus contains approximately **56 hours of orthographically transcribed broadband speech**. The official test suite contains **8 speakers**.

The corpus is not redistributed in this repository. Researchers must obtain it from its official source and comply with the original licence terms.

## Dataset reference

N. J. de Vries, M. H. Davel, J. Badenhorst, W. D. Basson, F. de Wet, E. Barnard, and A. de Waal, “A smartphone-based ASR data collection tool for under-resourced languages,” *Speech Communication*, vol. 56, pp. 119–131, 2014.

## Dataset preparation

The NCHLT XML transcription files are parsed and matched with their corresponding WAV recordings.

The preparation workflow used in this project includes:

- checking that referenced audio files exist;
- excluding missing or empty transcription entries;
- converting transcripts to lowercase;
- removing unnecessary punctuation;
- normalising repeated whitespace;
- preserving Northern Sotho characters such as `š`;
- standardising audio to **16 kHz**;
- keeping the official test partition isolated from model training and validation.

## Final documented split

| Partition | Utterances |
|---|---:|
| Training | 50,655 |
| Validation | 5,629 |
| Official test | 2,829 |
| **Total** | **59,113** |

The NCHLT development portion contains **56,284 utterances** before the internal training/validation split.

An **utterance** is one speech-audio recording paired with its corresponding orthographic transcription.

## Historical split note

The saved training notebook contains a one-utterance internal split difference:

- notebook output: 50,656 training / 5,628 validation;
- final submitted report: 50,655 training / 5,629 validation.

The development total remains 56,284 in both cases, and the official test set remains 2,829 utterances.

The repository therefore uses the final report counts as the documented reference configuration. Reconstructed manifests should not be described as proof of the exact historical row membership unless the original manifests are available.

## Local dataset layout

A typical local layout is:

```text
nchlt_nso/
├── audio/
│   ├── <speaker_id>/
│   └── ...
└── transcriptions/
    ├── nchlt_nso.trn.xml
    └── nchlt_nso.tst.xml
```

The exact extraction layout may vary. Dataset paths should be supplied to the scripts as command-line arguments rather than hard-coded into the repository.
