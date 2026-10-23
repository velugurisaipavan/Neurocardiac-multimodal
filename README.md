# AI-Enhanced Multimodal Framework for Early Diagnosis of Neuro-Cardiac Disorders

Implementation for **"AI-Enhanced Multimodal Framework for Early Diagnosis of Neuro-Cardiac Disorders"**
Sai Pavan Veluguri — *2025 IEEE 3rd Global Conference on Wireless Computing and Networking (GCWCN)* · 🏆 **Best Paper Award**

[![DOI](https://img.shields.io/badge/DOI-10.1109%2FGCWCN66157.2025.11448396-blue)](https://doi.org/10.1109/GCWCN66157.2025.11448396)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Research use only.** Not for diagnosis, screening, treatment, alerts, or deployment to patients. No patient data or trained weights are included; results come from training on your own paired cohort.

## Idea

Cardiac and neurological conditions often occur together, but most diagnostic models look at one signal at a time. This framework reads three sources together — **ECG**, **EEG**, and **clinical notes** — encodes each with a model suited to it, and uses **attention-weighted fusion** to learn how much each source should count for a given patient.

## Results reported in the paper

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| ECG only (CNN–BiLSTM) | 91.2% | — | — | — | — |
| EEG only (Attention-CNN) | 89.5% | — | — | — | 0.918 |
| Text only (ClinicalBERT) | 88.7% | — | — | — | 0.909 |
| Early-fusion concatenation | 93.6% | — | — | — | 0.941 |
| **Attention-based fusion (proposed)** | **97.3%** | **96.8%** | **97.6%** | **97.1%** | **0.985** |

Ablation — removing any one part lowers performance:

| Configuration | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Full model (ECG + EEG + text, attention fusion) | 97.3% | 96.8% | 97.6% | 97.1% | 0.985 |
| Without ECG | 92.5% | 91.4% | 92.1% | 91.7% | 0.936 |
| Without EEG | 93.1% | 92.8% | 92.4% | 92.6% | 0.941 |
| Without text | 93.6% | 93.1% | 93.0% | 93.0% | 0.944 |
| Without attention (simple concatenation) | 93.8% | 93.2% | 93.9% | 93.5% | 0.941 |

Results were reported with 10-fold cross-validation. The cohort is not public, so these figures come from the paper and are not shipped with this code.

## Architecture

```text
Aligned ECG window -> bandpass / resample / normalize -> CNN + BiLSTM ---+
                                                                       |
Aligned EEG window -> artifact-reviewed / STFT -> attention CNN --------+-> learned modality
                                                                       |   weights -> weighted
Pre-cutoff clinical note -> ClinicalBERT -> masked mean pooling --------+   sum -> MLP -> logit
                                                                                          |
                                                                                 sigmoid probability
```

All encoders project into the same latent dimension before attention-weighted fusion. The fusion layer learns a separate scalar attention score for each modality, normalizes scores across modalities using softmax, and forms a weighted sum. The model trains with binary cross-entropy-with-logits loss and Adam. Validation loss controls learning-rate scheduling and early stopping. Full ClinicalBERT fine-tuning is the default; `--freeze-text` is an explicit memory-saving alternative.

## Install

Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
pytest -q
```

CPU-only PyTorch can be installed first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

For optional continuous-recording EEG ICA:

```bash
pip install -e ".[ica]"
```

The actual pretrained text model downloads on the first text-enabled training run. Model identifier: `emilyalsentzer/Bio_ClinicalBERT`. The default revision is pinned to `d5892b39a4adaed74b92212a44081509db72f87b`; use `--text-revision` if you intentionally change it. Review the model card and applicable third-party terms:
https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT

Dependency ranges are not a frozen environment. Save `pip freeze > runs/your-run/environment.txt` after training for a complete package record. GPU execution requires an appropriate PyTorch/CUDA installation and is not validated here.

## Data you must provide

The paired ECG–EEG–clinical-text cohort is not public and is not included.

Provide legitimate, de-identified, temporally aligned records from the **same patient and encounter**, with independently adjudicated binary labels. Never join an unrelated ECG dataset, EEG dataset, and clinical notes by row number or class label. Suitable access permissions, research approvals, and a documented clinical endpoint are your responsibility.

Each raw `.npz` must contain:

- `ecg`: numeric array `[leads, time]`, default one lead
- `eeg`: numeric array `[electrodes, time]`, default 19 channels

Use the same channel identities/order, physical units, reference montage, acquisition convention, and time alignment throughout training and inference. Defaults assume complete 10-second windows. Lead/electrode names and montage are not automatically validated; keep a separate documented acquisition specification.

Each clinical note is a UTF-8 `.txt` file. Do not upload sensitive records to GitHub, external APIs, CI, or shared artifacts. Model/tokenizer downloads do not require uploading patient notes: text encoding runs locally.

## Manifest format

Start from `examples/manifest-template.csv`, which contains only a header and **no invented patient records**. Add one row per actual paired window:

| Column | Meaning |
|---|---|
| `sample_id` | Unique de-identified record/window identifier |
| `patient_id` | Stable de-identified patient identifier; cannot cross train/validation/test |
| `split` | `train`, `val`, or `test` |
| `label` | `0` or `1`, based on your predefined, adjudicated endpoint |
| `signal_path` | Raw `.npz` containing aligned ECG and EEG |
| `text_path` | UTF-8 note available by the prediction cutoff |
| `ecg_fs`, `eeg_fs` | Source sample rates in Hz, both above 90 for the configured 45-Hz low-pass edge |
| `window_start`, `window_end` | ISO-8601 timestamps with timezone; define the shared signal interval |
| `note_available_at` | Timestamp at which the note was actually available, not just the event date |
| `eeg_artifacts_reviewed` | `1` only after EEG artifact inspection/cleaning; otherwise preprocessing refuses to proceed |

Paths are relative to the manifest's folder unless absolute. The EEG artifact flag is a workflow assertion, not proof that data are clean. The code cannot inspect the note for diagnostic leakage or verify that raw signal timestamps match the manifest. Audit both yourself.

The prediction cutoff is **the end of the completed signal window**. Notes later than that are rejected. Future outcomes, retrospective discharge summaries, explicit target diagnoses, or post-event annotations can still leak labels even if their timestamps pass; remove them according to your endpoint protocol. To claim early prediction, define a future outcome horizon, record outcome times, enforce a lead time, and validate it separately. This code alone does not establish early-diagnosis capability.

Split patients before any cohort-level fitting. Use both classes in each split. Multiple windows of a patient stay in one split. Avoid duplicated encounters and cross-site proxies. Pretrained ClinicalBERT was trained on MIMIC notes; assess possible corpus/patient overlap if you evaluate on MIMIC-derived cohorts.

## Preprocess

```bash
python -m neurocardiac.preprocess \
  --manifest data/manifest.csv \
  --out-dir cache/paired \
  --duration 10 \
  --ecg-leads 1 \
  --eeg-channels 19
```

The preprocessing module checks channel counts, duration, finite values, cutoff timestamps, label validity, and patient split exclusivity. It bandpass-filters at 0.5–45 Hz, resamples ECG to 250 Hz and EEG to 128 Hz, standardizes each ECG channel, and computes EEG log-power STFT features with one-second segments and 50% overlap. EEG spectra are standardized per electrode over frequency/time. No cohort-level normalization is fitted.

The cache contains processed signals **and copies of clinical notes**. Keep it private. Input files are loaded with `allow_pickle=False`. Do not assume a pseudonymized cache is automatically anonymous.

Filtering uses forward-backward filtering inside a completed window, and the ECG encoder is bidirectional. This is **not** a strictly causal online streaming detector. No R-peak/beat segmentation is implemented: complete fixed-duration windows preserve ECG–EEG alignment instead. 

### Optional reviewed ICA

ICA should be inspected on a suitably long, continuous multichannel recording, not blindly fitted on every short window:

```bash
python -m neurocardiac.eeg_ica \
  --input-fif data/continuous-eeg-raw.fif \
  --out-dir data/ica-inspection
```

The inspect-only run saves `components-ica.fif` and `sources-raw.fif`. Open them in MNE and identify ocular/muscular artifacts using the actual recording, EOG channels when available, and expert inspection. Then rerun to a different directory, passing **your reviewed indices** via `--exclude`. No fixed artifact indices are supplied, because they vary by recording.

The helper fits ICA on a 1-Hz-high-passed copy, applies selected exclusions to the original EEG, and saves cleaned continuous EEG plus metadata. A dataset-specific adapter must then cut synchronized EEG and ECG windows and build the paired `.npz` files. This helper is optional. The default preprocessing command does not silently claim to perform ICA or automatically identify artifacts.

## Train

Full model:

```bash
python -m neurocardiac.train \
  --cache-dir cache/paired \
  --out-dir runs/full-attention \
  --epochs 30 \
  --batch-size 8
```

For GPU training, add `--device cuda` if supported. For lower memory, add `--freeze-text` and/or reduce `--batch-size`. Full fine-tuning updates the text model as well as the signal encoders and fusion/classifier layers.

The CLI uses the explicit splits in the manifest. It never selects hyperparameters or thresholds using the test set. The operating threshold is fixed at 0.5 for research comparison, not a clinically justified alert threshold. Checkpoints are selected only by validation BCE; the test set is evaluated after loading the best checkpoint. Existing nonempty output directories are rejected.

## Ablations

Train each configuration separately on identical splits:

```bash
# ECG-only CNN-BiLSTM
python -m neurocardiac.train --cache-dir cache/paired --modalities ecg --out-dir runs/ecg-only

# EEG-only attention CNN
python -m neurocardiac.train --cache-dir cache/paired --modalities eeg --out-dir runs/eeg-only

# Clinical-text only
python -m neurocardiac.train --cache-dir cache/paired --modalities text --out-dir runs/text-only

# Static concatenation of all three embeddings
python -m neurocardiac.train --cache-dir cache/paired --fusion concat --out-dir runs/concatenation

# No-text fusion
python -m neurocardiac.train --cache-dir cache/paired --modalities ecg eeg --out-dir runs/no-text
```

Two-modality variants can use `--modalities eeg text` or `--modalities ecg text`. These are retrained ablations, not masks applied to an already fitted full model. All variants still use the same paired cohort for fair comparison. Automatically running 10-fold cross-validation, paired statistical tests, patient-level aggregation, calibration, SHAP/LIME, and bootstrap confidence intervals are not included. A single split is not directly comparable with the paper's 10-fold results.

## Inference without labels

For new paired records, omit `label` and `split` from the raw inference manifest. All timing, identity, sampling-rate, and EEG-review columns remain required.

Use the preprocessing configuration from the training cache:

```bash
python -m neurocardiac.preprocess \
  --manifest data/inference-manifest.csv \
  --out-dir cache/inference \
  --unlabeled \
  --config cache/paired/preprocessing.json
```

If the training cache is unavailable, extract its preprocessing settings from the saved run first:

```bash
python -c "import json,pathlib; p=pathlib.Path('runs/full-attention'); (p/'config-preprocessing.json').write_text(json.dumps(json.loads((p/'config.json').read_text())['preprocessing']))"
```

Then use `--config runs/full-attention/config-preprocessing.json` instead. Run saved-model inference:

```bash
python -m neurocardiac.predict \
  --run-dir runs/full-attention \
  --cache-dir cache/inference \
  --output runs/full-attention/inference.csv
```

Inference reads no target labels and checks that preprocessing matches training. The saved text config, tokenizer, and model weights are sufficient for local inference without redownloading ClinicalBERT. Only load trusted model artifacts. Inference is CPU-only in this CLI.

## Outputs and interpretation

Successful training writes:

- `model.pt`: learned model state, including fine-tuned or frozen text weights where applicable
- `config.json`: settings, split counts, preprocessing, best epoch, and key package versions
- `training.csv`: observed training/validation losses and validation AUC
- `metrics.json`: observed held-out accuracy, precision, sensitivity, specificity, F1, ROC AUC, average precision, Brier score, and confusion matrix
- `predictions.csv`: actual record-level predictions and learned modality attention weights
- `tokenizer/` and `text_config/`: saved local text configuration for text-enabled runs

Metric values are fractions, not percentages. Metrics are computed at the **paired-window/record level**, not automatically at patient level. Patients with more windows contribute more weight; define and validate patient-level aggregation separately if needed. Repeated windows are not independent observations for confidence intervals or significance testing.

A sigmoid value is an **uncalibrated research classifier output**, not a validated individual probability of disease. Attention weights describe internal routing, not causal importance, reliable clinical explanations, or proof that a modality caused a prediction. They are not a substitute for SHAP, saliency validation, calibration, subgroup audits, external validation, or clinician review. No normal/moderate/high clinical risk tiers or automatic alerts are implemented.

## Implementation notes

| Component | Implemented | Notes |
|---|---|---|
| ECG branch | 1D CNN + BiLSTM | Fixed aligned windows; no R-peak segmentation |
| EEG branch | Log-power STFT + convolutional spatial attention | ICA is a separate reviewed workflow |
| Text branch | Bio_ClinicalBERT + mask-aware mean pooling | Native tokenizer retains negation/context, no stop-word deletion or lemmatization |
| Fusion | Modality-specific softmax scores and weighted sum in common latent space | Projection and scorer sizes configurable |
| Classifier | MLP and sigmoid at inference | No clinically validated risk threshold or calibration |
| Training | BCE-with-logits, Adam, dropout, scheduling, early stopping | Configurable defaults |
| Evaluation | Explicit patient-disjoint train/val/test; real computed metrics only | No bundled dataset; no 10-fold runner |
| Interpretation | Exported learned modality weights | No SHAP/LIME, causal interpretation, or verified clinical explanatory power |

## Repository layout

```text
neurocardiac/
  manifest.py     # Patient/time/label checks
  preprocess.py   # Paired-window filtering, resampling, EEG STFT
  eeg_ica.py      # Optional reviewed continuous EEG ICA helper
  model.py        # Three encoders, attention fusion, classifier
  data.py         # Paired cache and mask-aware tokenization
  train.py        # Validation-selected training and held-out metrics
  predict.py      # Unlabeled saved-model inference
  metrics.py      # Observed binary metrics
examples/
  manifest-template.csv  # Header only; no fabricated patient data
tests/
  test_pipeline.py      # Software checks, not clinical validation
```

## Citation

```bibtex
@inproceedings{veluguri2025neurocardiac,
  author    = {Veluguri, Sai Pavan},
  title     = {AI-Enhanced Multimodal Framework for Early Diagnosis of Neuro-Cardiac Disorders},
  booktitle = {2025 IEEE 3rd Global Conference on Wireless Computing and Networking (GCWCN)},
  year      = {2025},
  publisher = {IEEE},
  doi       = {10.1109/GCWCN66157.2025.11448396}
}
```

## License

Code is released under the [MIT License](LICENSE). The paper is © IEEE and is not bundled here; read it on [IEEE Xplore](https://doi.org/10.1109/GCWCN66157.2025.11448396).
