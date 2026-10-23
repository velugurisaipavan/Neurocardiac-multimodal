# Validation status

## Executed checks

- 17 unit/smoke tests passed in the generation environment.
- All source/test modules passed Python compilation.
- The preprocess, train, predict, and eeg_ica command-line help checks passed.
- ECG, EEG, and injected text branches completed forward/backward passes; modality ablations and concatenation were checked.
- Attention weights normalized over the selected modalities, and masked text pooling excluded padding.
- A tiny locally initialized BERT configuration checked the actual transformer wrapper and its frozen mode without downloading pretrained weights.
- Saved full three-modality model/tokenizer inference passed using that tiny local configuration and software fixtures.
- One-epoch two-signal-branch training, checkpoint selection, metric writing, and saved-model inference passed using deterministic software fixtures.
- Manifest checks rejected cross-split patients, future notes, and unreviewed EEG.

## Not established

No paired clinical cohort was supplied or downloaded. No actual ClinicalBERT pretrained-weight download, full real-cohort training, reported-paper result reproduction, clinical validation, real EEG artifact correction, GPU execution, calibrated patient risk, timing benchmark, or external validation was performed. The optional MNE ICA helper was not executed. Tiny transformer and zero-array fixtures are only software tests; their outputs are not physiological observations or accuracy evidence. Test training outputs were confined to temporary directories and are not bundled.

One environment warning concerned the installed requests/urllib3/charset-normalizer versions; it did not fail these tests. See pytest output if reproducing in your environment. Dependency ranges, rather than a complete environment lockfile, are supplied.

## Generation environment

- Python: 3.12.8
- numpy: 1.26.4
- scipy: 1.13.1
- scikit-learn: 1.6.1
- torch: 2.14.1+cpu
- transformers: 4.51.3
- pytest: 8.3.5
