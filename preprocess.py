"""Offline paired-window processing. No automatic pairing of unrelated datasets."""
import argparse
import csv
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt, stft

from .manifest import load_manifest

DEFAULT_CONFIG = dict(duration_seconds=10.0, ecg_fs=250, eeg_fs=128, ecg_leads=1, eeg_channels=19,
                      low_hz=0.5, high_hz=45.0, filter_order=4, stft_seconds=1.0, stft_overlap=0.5)


def as_channels(values):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Expected finite signal [channels, time]")
    return x


def standardize(x, axes):
    return (x - x.mean(axis=axes, keepdims=True)) / np.maximum(x.std(axis=axes, keepdims=True), 1e-6)


def filter_and_resample(x, source_fs, target_fs, config):
    if source_fs <= 2 * config["high_hz"] or target_fs <= 2 * config["high_hz"]:
        raise ValueError("Nyquist frequency must exceed the low-pass edge")
    sos = butter(config["filter_order"], [config["low_hz"], config["high_hz"]],
                 btype="bandpass", fs=source_fs, output="sos")
    # Noncausal forward-backward filtering: suitable for completed offline windows.
    x = sosfiltfilt(sos, x, axis=-1)
    ratio = (Fraction(str(target_fs)) / Fraction(str(source_fs))).limit_denominator(100000)
    return resample_poly(x, ratio.numerator, ratio.denominator, axis=-1)


def process_arrays(ecg, eeg, ecg_fs, eeg_fs, config):
    ecg, eeg = as_channels(ecg), as_channels(eeg)
    for name, x, fs, channels in [("ECG", ecg, ecg_fs, config["ecg_leads"]),
                                   ("EEG", eeg, eeg_fs, config["eeg_channels"])]:
        if x.shape[0] != channels:
            raise ValueError(f"{name}: wrong channel count or transposed array")
        if abs(x.shape[1] / fs - config["duration_seconds"]) > 1.01 / fs:
            raise ValueError(f"{name}: supply a time-aligned window of configured duration")
    ecg = filter_and_resample(ecg, ecg_fs, config["ecg_fs"], config)
    eeg = filter_and_resample(eeg, eeg_fs, config["eeg_fs"], config)
    ecg_size = round(config["duration_seconds"] * config["ecg_fs"])
    eeg_size = round(config["duration_seconds"] * config["eeg_fs"])
    if ecg.shape[1] < ecg_size or eeg.shape[1] < eeg_size:
        raise ValueError("Resampled signal is shorter than expected")
    ecg = standardize(ecg[:, :ecg_size], (-1,)).astype(np.float32)
    eeg = eeg[:, :eeg_size]
    nperseg = round(config["stft_seconds"] * config["eeg_fs"])
    if nperseg > eeg_size:
        raise ValueError("Window shorter than STFT segment")
    frequencies, _, spectrum = stft(eeg, fs=config["eeg_fs"], nperseg=nperseg,
                                   noverlap=round(nperseg * config["stft_overlap"]),
                                   axis=-1, boundary=None, padded=False)
    keep = (frequencies >= config["low_hz"]) & (frequencies <= config["high_hz"])
    log_power = np.log1p(np.abs(spectrum[:, keep, :]) ** 2)
    eeg_spectrum = standardize(log_power, (-2, -1)).astype(np.float32)
    if not (np.isfinite(ecg).all() and np.isfinite(eeg_spectrum).all()):
        raise ValueError("Nonfinite processed signals")
    return ecg, eeg_spectrum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Use training preprocessing.json for inference")
    parser.add_argument("--unlabeled", action="store_true")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--ecg-leads", type=int, default=1)
    parser.add_argument("--eeg-channels", type=int, default=19)
    args = parser.parse_args()
    config = json.loads(args.config.read_text()) if args.config else {
        **DEFAULT_CONFIG, "duration_seconds": args.duration, "ecg_leads": args.ecg_leads, "eeg_channels": args.eeg_channels}
    if config["duration_seconds"] < 2 or min(config["ecg_leads"], config["eeg_channels"]) < 1:
        raise ValueError("Duration >=2 seconds and positive channel counts required")
    rows = load_manifest(args.manifest, labeled=not args.unlabeled)
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise ValueError("Choose an empty output directory")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for i, row in enumerate(rows):
        if abs(row["duration_seconds"] - config["duration_seconds"]) > 1e-3:
            raise ValueError("Manifest duration differs from preprocessing config")
        with np.load(row["signal_path"], allow_pickle=False) as signals:
            ecg, eeg = process_arrays(signals["ecg"], signals["eeg"], row["ecg_fs"], row["eeg_fs"], config)
        text = Path(row["text_path"]).read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError("Clinical note is empty")
        name = f"record_{i:07d}"
        np.savez_compressed(args.out_dir / f"{name}.npz", ecg=ecg, eeg=eeg)
        (args.out_dir / f"{name}.txt").write_text(text, encoding="utf-8")
        record = {k: row[k] for k in ("sample_id", "patient_id", "window_end")}
        record.update(signal_path=f"{name}.npz", text_path=f"{name}.txt")
        if not args.unlabeled:
            record.update(split=row["split"], label=row["label"])
        records.append(record)
    with (args.out_dir / "records.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    config_text = json.dumps(config, indent=2)
    (args.out_dir / "preprocessing.json").write_text(config_text)
    (args.out_dir / "provenance.json").write_text(json.dumps({
        "source_manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "records": len(records), "original_experimental_code": False}, indent=2))
    print(f"Prepared {len(records)} paired records. Cache includes sensitive notes; keep it private.")


if __name__ == "__main__":
    main()
