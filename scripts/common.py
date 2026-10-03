from __future__ import annotations

import re
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath
from typing import Iterable

import pandas as pd


def normalize_text(text: str) -> str:
    """
    Apply the transcript normalisation used by the verified experiment.

    Policy:
    1. Unicode NFC normalisation.
    2. Lowercase.
    3. Preserve Unicode letters and combining marks (for example ``š``).
    4. Replace punctuation, digits, symbols, and control characters with spaces.
    5. Collapse repeated whitespace.

    This intentionally does *not* use ``\\w`` because ``\\w`` also preserves
    digits and underscores, which were not part of the executed text-cleaning
    policy.
    """
    text = unicodedata.normalize("NFC", str(text)).lower()
    cleaned: list[str] = []

    for character in text:
        category = unicodedata.category(character)
        if category.startswith("L") or category.startswith("M"):
            cleaned.append(character)
        elif character.isspace():
            cleaned.append(" ")
        else:
            cleaned.append(" ")

    return " ".join("".join(cleaned).split())


def _local_tag(tag: str) -> str:
    """Return an XML tag name without an optional namespace prefix."""
    return tag.split("}")[-1]


def parse_nchlt_xml(xml_path: Path, dataset_root: Path) -> pd.DataFrame:
    """
    Parse an NCHLT transcription XML file into an utterance-level manifest.

    ``dataset_root`` is the directory that contains the top-level ``nchlt_nso``
    folder. XML audio references are kept as portable relative paths while an
    absolute local ``audio_path`` is also written for downstream scripts.
    """
    xml_path = Path(xml_path).resolve()
    dataset_root = Path(dataset_root).resolve()

    if not xml_path.is_file():
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    rows: list[dict] = []
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for speaker in root.iter():
        if _local_tag(speaker.tag) != "speaker":
            continue

        speaker_id = str(speaker.attrib.get("id", "")).strip()
        age = str(speaker.attrib.get("age", "")).strip()
        gender = str(speaker.attrib.get("gender", "")).strip()
        location = str(speaker.attrib.get("location", "")).strip()

        for recording in list(speaker):
            if _local_tag(recording.tag) != "recording":
                continue

            audio_rel = str(recording.attrib.get("audio", "")).strip()
            if not audio_rel:
                continue

            relpath = PurePosixPath(audio_rel)
            if relpath.is_absolute() or ".." in relpath.parts:
                raise ValueError(f"Unsafe audio path in XML: {audio_rel}")

            audio_path = dataset_root.joinpath(*relpath.parts).resolve()
            if not audio_path.is_relative_to(dataset_root):
                raise ValueError(f"Audio path escapes dataset root: {audio_path}")

            orth_nodes = [
                node for node in recording.iter()
                if _local_tag(node.tag) == "orth"
            ]
            reference = ""
            if orth_nodes:
                reference = " ".join(
                    "".join(orth_nodes[0].itertext()).split()
                )

            duration = recording.attrib.get("duration")

            rows.append(
                {
                    "speaker": speaker_id,
                    "age": age,
                    "gender": gender,
                    "location": location,
                    "audio_relpath": relpath.as_posix(),
                    "filename": relpath.name,
                    "audio_path": str(audio_path),
                    "duration": float(duration) if duration else None,
                    "reference": reference,
                    "reference_norm": normalize_text(reference),
                }
            )

    return pd.DataFrame(rows)


def audit_manifest(df: pd.DataFrame, check_audio: bool = True) -> dict:
    """Return the integrity checks used before any split is created."""
    missing_audio = 0
    if check_audio:
        missing_audio = int(
            (~df["audio_path"].map(lambda value: Path(value).is_file())).sum()
        )

    duplicate_key = "audio_relpath" if "audio_relpath" in df else "filename"

    return {
        "utterances": int(len(df)),
        "unique_speakers": int(df["speaker"].nunique()) if "speaker" in df else None,
        "missing_audio_files": missing_audio,
        "empty_transcripts": int(df["reference_norm"].eq("").sum()),
        "duplicate_audio_paths": int(df[duplicate_key].duplicated().sum()),
        "reference_word_tokens": int(
            df["reference_norm"].str.split().str.len().fillna(0).sum()
        ),
    }


def ensure_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required CSV columns: {missing}")
