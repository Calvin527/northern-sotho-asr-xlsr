from __future__ import annotations

import re
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

import pandas as pd


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text))
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_nchlt_xml(xml_path: Path, dataset_root: Path) -> pd.DataFrame:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    rows = []
    for speaker in root.findall("speaker"):
        speaker_id = speaker.attrib.get("id", "")
        age = speaker.attrib.get("age", "")
        gender = speaker.attrib.get("gender", "")
        location = speaker.attrib.get("location", "")
        for recording in speaker.findall("recording"):
            audio_rel = recording.attrib.get("audio")
            if not audio_rel:
                continue
            orth = recording.find("orth")
            reference = orth.text.strip() if orth is not None and orth.text else ""
            duration = recording.attrib.get("duration")
            audio_path = dataset_root / audio_rel
            rows.append({
                "speaker": speaker_id,
                "age": age,
                "gender": gender,
                "location": location,
                "filename": Path(audio_rel).name,
                "audio_path": str(audio_path),
                "duration": float(duration) if duration else None,
                "reference": reference,
                "reference_norm": normalize_text(reference),
            })
    return pd.DataFrame(rows)


def audit_manifest(df: pd.DataFrame, check_audio: bool = True) -> dict:
    missing_audio = 0
    if check_audio:
        missing_audio = int((~df["audio_path"].map(lambda x: Path(x).exists())).sum())
    return {
        "utterances": int(len(df)),
        "unique_speakers": int(df["speaker"].nunique()) if "speaker" in df else None,
        "missing_audio_files": missing_audio,
        "empty_transcripts": int(df["reference_norm"].eq("").sum()),
        "duplicate_filenames": int(df["filename"].duplicated().sum()),
        "reference_word_tokens": int(df["reference_norm"].str.split().str.len().sum()),
    }


def ensure_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required CSV columns: {missing}")
