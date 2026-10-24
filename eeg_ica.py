"""Optional offline ICA on a continuous recording with human-reviewed components."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-fif", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--exclude", type=int, nargs="+", default=[], help="Reviewed ICA component indices; none for inspect-only")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    import mne
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise ValueError("Use an empty output directory")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw = mne.io.read_raw_fif(args.input_fif, preload=True).pick("eeg")
    if len(raw.ch_names) < 2:
        raise ValueError("ICA requires multichannel EEG")
    fitting = raw.copy().filter(l_freq=1.0, h_freq=None)
    ica = mne.preprocessing.ICA(n_components=0.99, method="fastica", random_state=args.seed, max_iter=1000)
    ica.fit(fitting)
    ica.save(args.out_dir / "components-ica.fif", overwrite=True)
    ica.get_sources(fitting).save(args.out_dir / "sources-raw.fif", overwrite=True)
    if any(i < 0 or i >= ica.n_components_ for i in args.exclude):
        raise ValueError("ICA component index out of range")
    if args.exclude:
        ica.exclude = args.exclude
        cleaned = ica.apply(raw.copy())
        cleaned.save(args.out_dir / "cleaned-raw.fif", overwrite=True)
        np.save(args.out_dir / "cleaned_eeg.npy", cleaned.get_data())
        ica.save(args.out_dir / "components-ica.fif", overwrite=True)
    info = dict(sampling_rate=float(raw.info["sfreq"]), channel_order=raw.ch_names,
                excluded_components=args.exclude, status="cleaned-needs-QC" if args.exclude else "inspect-only")
    (args.out_dir / "metadata.json").write_text(json.dumps(info, indent=2))
    print("Review components and residual artifacts; no automatic clinical artifact identification is performed.")


if __name__ == "__main__":
    main()
