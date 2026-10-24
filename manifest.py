"""Strict input contracts: explicit patient-disjoint splits and pre-cutoff notes."""
import csv
from datetime import datetime, timezone
from pathlib import Path


def timestamp(value):
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise ValueError("Timestamps require explicit timezone")
    return stamp.astimezone(timezone.utc)


def load_manifest(path, labeled=True):
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Manifest is empty; supply actual paired records")
    required = {"sample_id", "patient_id", "signal_path", "text_path", "ecg_fs", "eeg_fs",
                "window_start", "window_end", "note_available_at", "eeg_artifacts_reviewed"}
    if labeled:
        required |= {"split", "label"}
    if required - set(rows[0]):
        raise ValueError(f"Missing manifest columns: {sorted(required - set(rows[0]))}")
    ids, patients = set(), {}
    for row in rows:
        if not row["sample_id"] or row["sample_id"] in ids or not row["patient_id"]:
            raise ValueError("Need unique nonempty sample_id and nonempty patient_id")
        ids.add(row["sample_id"])
        start, end, note = [timestamp(row[k]) for k in ("window_start", "window_end", "note_available_at")]
        if end <= start or note > end:
            raise ValueError("Invalid window or note unavailable at prediction cutoff (window_end)")
        row["duration_seconds"] = (end - start).total_seconds()
        for key in ("ecg_fs", "eeg_fs"):
            rate = float(row[key])
            if not 90 < rate <= 10000:
                raise ValueError("Sampling rates must exceed 90 Hz for the configured 45-Hz low pass")
            row[key] = rate
        if row["eeg_artifacts_reviewed"] != "1":
            raise ValueError("EEG artifact review/cleaning must be completed; do not just set this flag")
        if labeled:
            if row["split"] not in {"train", "val", "test"} or row["label"] not in {"0", "1"}:
                raise ValueError("Expected split train/val/test and binary labels 0/1")
            row["label"] = int(row["label"])
            previous = patients.setdefault(row["patient_id"], row["split"])
            if previous != row["split"]:
                raise ValueError(f"Patient crosses splits: {row['patient_id']}")
        for key in ("signal_path", "text_path"):
            source = Path(row[key])
            source = source if source.is_absolute() else path.parent / source
            if not source.is_file():
                raise ValueError(f"Missing {key}: {source}")
            row[key] = str(source.resolve())
    return rows
