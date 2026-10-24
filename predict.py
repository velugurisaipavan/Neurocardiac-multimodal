"""Unlabeled inference; no ground-truth labels are consumed."""
import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import Collator, PairedDataset, model_inputs, read_records
from .model import NeuroCardiacModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    config = json.loads((args.run_dir / "config.json").read_text())
    rows, preprocessing = read_records(args.cache_dir, labeled=False)
    if preprocessing != config["preprocessing"]:
        raise ValueError("Training and inference preprocessing differ")
    tokenizer = None
    if "text" in config["modalities"]:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.run_dir / "tokenizer", local_files_only=True)
    model = NeuroCardiacModel(ecg_leads=preprocessing["ecg_leads"], eeg_channels=preprocessing["eeg_channels"],
                             dim=config["dim"], dropout=config["dropout"], modalities=config["modalities"],
                             fusion=config["fusion"], text_config_dir=args.run_dir / "text_config")
    model.load_state_dict(torch.load(args.run_dir / "model.pt", map_location="cpu", weights_only=True))
    model.eval()
    loader = DataLoader(PairedDataset(args.cache_dir, rows, labeled=False), batch_size=args.batch_size,
                        collate_fn=Collator(tokenizer, config["max_tokens"], labeled=False))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream, torch.no_grad():
        writer = csv.writer(stream)
        attention_columns = [f"attention_{m}" for m in config["modalities"]] if config["fusion"] == "attention" else []
        writer.writerow(["sample_id", "classifier_probability", "predicted_class"] + attention_columns)
        for batch in loader:
            logits, weights = model(**model_inputs(batch, "cpu"))
            probabilities = logits.sigmoid().tolist()
            for i, p in enumerate(probabilities):
                writer.writerow([batch["sample_id"][i], p, int(p >= 0.5)] + (weights[i].tolist() if weights is not None else []))


if __name__ == "__main__":
    main()
