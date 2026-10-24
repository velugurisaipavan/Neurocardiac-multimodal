"""Train one fixed patient-disjoint split with validation-only checkpoint selection."""
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import Collator, PairedDataset, model_inputs, read_records
from .metrics import binary_metrics
from .model import NeuroCardiacModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--modalities", nargs="+", choices=["ecg", "eeg", "text"], default=["ecg", "eeg", "text"])
    parser.add_argument("--fusion", choices=["attention", "concat"], default="attention")
    parser.add_argument("--text-model", default="emilyalsentzer/Bio_ClinicalBERT")
    parser.add_argument("--text-revision", default="d5892b39a4adaed74b92212a44081509db72f87b")
    parser.add_argument("--freeze-text", action="store_true")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.patience, args.dim) < 1 or not 2 <= args.max_tokens <= 512:
        raise ValueError("Invalid positive hyperparameters or max-tokens outside 2..512")
    if args.learning_rate <= 0 or not 0 <= args.dropout < 1:
        raise ValueError("Invalid learning rate or dropout")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise ValueError("Use an empty output directory")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    records, preprocessing = read_records(args.cache_dir)
    splits = {split: [r for r in records if r["split"] == split] for split in ("train", "val", "test")}
    for name, rows in splits.items():
        if not rows or {r["label"] for r in rows} != {"0", "1"}:
            raise ValueError(f"{name} requires both classes; establish a valid patient-level split first")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(preprocessing=preprocessing, implementation="new paper-aligned reconstruction",
                  split_counts={k: len(v) for k, v in splits.items()},
                  patient_counts={k: len({r['patient_id'] for r in v}) for k, v in splits.items()},
                  versions=dict(numpy=np.__version__, torch=str(torch.__version__)))
    tokenizer = None
    if "text" in args.modalities:
        from transformers import AutoTokenizer, __version__ as transformers_version
        tokenizer = AutoTokenizer.from_pretrained(args.text_model, revision=args.text_revision)
        tokenizer.save_pretrained(args.out_dir / "tokenizer")
        config["versions"]["transformers"] = transformers_version
    collator = Collator(tokenizer, args.max_tokens)
    loaders = {k: DataLoader(PairedDataset(args.cache_dir, v), batch_size=args.batch_size, shuffle=k == "train",
                              collate_fn=collator, num_workers=0) for k, v in splits.items()}
    model = NeuroCardiacModel(ecg_leads=preprocessing["ecg_leads"], eeg_channels=preprocessing["eeg_channels"],
                             dim=args.dim, dropout=args.dropout, modalities=args.modalities, fusion=args.fusion,
                             text_model=args.text_model, freeze_text=args.freeze_text, text_revision=args.text_revision).to(device)
    if "text" in args.modalities:
        model.text_encoder.transformer.config.save_pretrained(args.out_dir / "text_config")
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    criterion = torch.nn.BCEWithLogitsLoss()

    def evaluate(loader):
        model.eval()
        total, ids, patients, labels, probabilities, weights = 0.0, [], [], [], [], []
        with torch.no_grad():
            for batch in loader:
                logits, attention = model(**model_inputs(batch, device))
                y = batch["label"].to(device)
                total += criterion(logits, y).item() * len(y)
                ids.extend(batch["sample_id"])
                patients.extend(batch["patient_id"])
                labels.extend(y.cpu().tolist())
                probabilities.extend(logits.sigmoid().cpu().tolist())
                if attention is not None:
                    weights.extend(attention.cpu().tolist())
        return total / len(labels), ids, patients, labels, probabilities, weights

    best, stale, log = float("inf"), 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_sum, count = 0.0, 0
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(**model_inputs(batch, device))
            y = batch["label"].to(device)
            loss = criterion(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_sum += loss.item() * len(y)
            count += len(y)
        validation_loss, _, _, y_val, p_val, _ = evaluate(loaders["val"])
        scheduler.step(validation_loss)
        log.append(dict(epoch=epoch, train_loss=train_sum / count, validation_loss=validation_loss,
                        validation_auc=binary_metrics(y_val, p_val)["roc_auc"], learning_rate=optimizer.param_groups[0]["lr"]))
        print(f"Epoch {epoch:03d}: validation BCE={validation_loss:.5f}")
        if validation_loss < best:
            best, stale = validation_loss, 0
            config["best_epoch"] = epoch
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, args.out_dir / "model.pt")
        else:
            stale += 1
            if stale >= args.patience:
                break
    if not (args.out_dir / "model.pt").exists():
        raise RuntimeError("No finite validation checkpoint")
    model.load_state_dict(torch.load(args.out_dir / "model.pt", weights_only=True, map_location=device))
    loss, ids, patients, labels, probabilities, weights = evaluate(loaders["test"])
    metrics = binary_metrics(labels, probabilities)
    metrics.update(test_bce=loss, unit="paired window/record", clinically_validated=False,
                   interpretation="Uncalibrated research classifier outputs, not validated individual clinical risk")
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2))
    with (args.out_dir / "training.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(log[0]))
        writer.writeheader()
        writer.writerows(log)
    with (args.out_dir / "predictions.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        attention_columns = [f"attention_{m}" for m in args.modalities] if weights else []
        writer.writerow(["sample_id", "patient_id", "label", "classifier_probability", "predicted_class"] + attention_columns)
        for i, p in enumerate(probabilities):
            writer.writerow([ids[i], patients[i], int(labels[i]), p, int(p >= 0.5)] + (weights[i] if weights else []))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
