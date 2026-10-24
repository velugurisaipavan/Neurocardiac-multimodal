"""Prepared paired records and local text tokenization."""
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def read_records(root, labeled=True):
    root = Path(root)
    with (root / "records.csv").open(newline="") as stream:
        records = list(csv.DictReader(stream))
    if not records:
        raise ValueError("No prepared records")
    patients, samples = {}, set()
    for row in records:
        if not row["sample_id"] or row["sample_id"] in samples or not row["patient_id"]:
            raise ValueError("Invalid patient/sample identifiers")
        samples.add(row["sample_id"])
        if labeled:
            if row.get("split") not in ("train", "val", "test") or row.get("label") not in ("0", "1"):
                raise ValueError("Invalid prepared label/split")
            previous = patients.setdefault(row["patient_id"], row["split"])
            if previous != row["split"]:
                raise ValueError("Patient crosses splits")
    return records, json.loads((root / "preprocessing.json").read_text())


class PairedDataset(Dataset):
    def __init__(self, root, records, labeled=True):
        self.root, self.records, self.labeled = Path(root), records, labeled

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        with np.load(self.root / row["signal_path"], allow_pickle=False) as signal:
            ecg, eeg = signal["ecg"].copy(), signal["eeg"].copy()
        if not (np.isfinite(ecg).all() and np.isfinite(eeg).all()):
            raise ValueError("Nonfinite cached signals")
        return dict(ecg=torch.from_numpy(ecg).float(), eeg=torch.from_numpy(eeg).float(),
                    text=(self.root / row["text_path"]).read_text(encoding="utf-8"),
                    label=float(row["label"]) if self.labeled else None,
                    sample_id=row["sample_id"], patient_id=row["patient_id"])


class Collator:
    def __init__(self, tokenizer=None, max_tokens=256, labeled=True):
        self.tokenizer, self.max_tokens, self.labeled = tokenizer, max_tokens, labeled

    def __call__(self, records):
        batch = {key: torch.stack([r[key] for r in records]) for key in ("ecg", "eeg")}
        if self.tokenizer is not None:
            tokens = self.tokenizer([r["text"] for r in records], padding=True, truncation=True,
                                    max_length=self.max_tokens, return_tensors="pt")
            batch.update(input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"])
        batch.update(sample_id=[r["sample_id"] for r in records], patient_id=[r["patient_id"] for r in records])
        if self.labeled:
            batch["label"] = torch.tensor([r["label"] for r in records], dtype=torch.float32)
        return batch


def model_inputs(batch, device):
    return {key: batch[key].to(device) for key in ("ecg", "eeg", "input_ids", "attention_mask") if key in batch}
