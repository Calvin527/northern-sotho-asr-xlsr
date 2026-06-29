# GitHub Upload Guide

## Do Upload

- Source code
- Notebooks
- README
- requirements.txt
- Methodology figures and tables
- Results summaries
- Small CSV files

## Do Not Upload

- NCHLT raw dataset
- WAV audio files
- Large model ZIP files
- `model.safetensors`
- Checkpoint folders
- Kaggle temporary working files

## Commands

```bash
git init
git add .
git commit -m "Initial commit: Northern Sotho ASR using XLS-R 300M"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/northern-sotho-asr-xlsr.git
git push -u origin main
```
