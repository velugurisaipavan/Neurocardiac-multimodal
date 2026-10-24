"""CNN-BiLSTM ECG + attention-CNN EEG + ClinicalBERT + attention-weighted fusion."""
import torch
from torch import nn

MODALITIES = ("ecg", "eeg", "text")


class ECGEncoder(nn.Module):
    def __init__(self, leads, dim):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv1d(leads, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
                                 nn.MaxPool1d(4), nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool1d(4))
        self.lstm = nn.LSTM(64, 64, batch_first=True, bidirectional=True)
        self.projection = nn.Linear(128, dim)

    def forward(self, x):
        sequence = self.cnn(x).transpose(1, 2)
        _, (hidden, _) = self.lstm(sequence)
        return self.projection(torch.cat([hidden[-2], hidden[-1]], dim=-1))


class EEGEncoder(nn.Module):
    def __init__(self, channels, dim):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv2d(channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                                 nn.MaxPool2d(2), nn.Conv2d(32, 64, 3, padding=1), nn.ReLU())
        self.attention = nn.Conv2d(64, 1, 1)
        self.projection = nn.Linear(64, dim)

    def forward(self, x):
        features = self.cnn(x)
        weights = self.attention(features).flatten(2).softmax(dim=-1)
        pooled = (features.flatten(2) * weights).sum(dim=-1)
        return self.projection(pooled)


def masked_mean(hidden, mask):
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)


class ClinicalTextEncoder(nn.Module):
    def __init__(self, model_name, dim, freeze=False, config_dir=None, revision=None):
        super().__init__()
        from transformers import AutoConfig, AutoModel
        if config_dir is not None:
            self.transformer = AutoModel.from_config(AutoConfig.from_pretrained(config_dir, local_files_only=True))
        else:
            self.transformer = AutoModel.from_pretrained(model_name, revision=revision)
        self.freeze = freeze
        if freeze:
            self.transformer.requires_grad_(False)
        self.projection = nn.Linear(self.transformer.config.hidden_size, dim)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.transformer.eval()
        return self

    def forward(self, ids, mask):
        if self.freeze:
            with torch.no_grad():
                output = self.transformer(input_ids=ids, attention_mask=mask).last_hidden_state
        else:
            output = self.transformer(input_ids=ids, attention_mask=mask).last_hidden_state
        return self.projection(masked_mean(output, mask))


class AttentionWeightedFusion(nn.Module):
    def __init__(self, count, dim):
        super().__init__()
        self.scorers = nn.ModuleList([nn.Linear(dim, 1) for _ in range(count)])

    def forward(self, features):
        scores = torch.cat([score(feature) for score, feature in zip(self.scorers, features)], dim=1)
        weights = scores.softmax(dim=1)
        fused = (torch.stack(features, dim=1) * weights.unsqueeze(-1)).sum(dim=1)
        return fused, weights


class NeuroCardiacModel(nn.Module):
    def __init__(self, ecg_leads=1, eeg_channels=19, dim=128, dropout=0.3,
                 modalities=MODALITIES, fusion="attention", text_encoder=None,
                 text_model="emilyalsentzer/Bio_ClinicalBERT", freeze_text=False,
                 text_config_dir=None, text_revision=None):
        super().__init__()
        self.modalities = tuple(modalities)
        if not self.modalities or len(set(self.modalities)) != len(self.modalities) or any(m not in MODALITIES for m in self.modalities):
            raise ValueError("Choose a nonempty, unique subset of ecg,eeg,text")
        if fusion not in ("attention", "concat"):
            raise ValueError("fusion must be attention or concat")
        self.fusion_kind = fusion
        if "ecg" in self.modalities:
            self.ecg_encoder = ECGEncoder(ecg_leads, dim)
        if "eeg" in self.modalities:
            self.eeg_encoder = EEGEncoder(eeg_channels, dim)
        if "text" in self.modalities:
            self.text_encoder = text_encoder if text_encoder is not None else ClinicalTextEncoder(
                text_model, dim, freeze_text, text_config_dir, text_revision)
        self.fusion = AttentionWeightedFusion(len(self.modalities), dim)
        size = dim if fusion == "attention" else dim * len(self.modalities)
        self.classifier = nn.Sequential(nn.Linear(size, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, ecg=None, eeg=None, input_ids=None, attention_mask=None):
        features = []
        for modality in self.modalities:
            if modality == "ecg":
                features.append(self.ecg_encoder(ecg))
            elif modality == "eeg":
                features.append(self.eeg_encoder(eeg))
            else:
                features.append(self.text_encoder(input_ids, attention_mask))
        if self.fusion_kind == "attention":
            fused, weights = self.fusion(features)
        else:
            fused, weights = torch.cat(features, dim=1), None
        return self.classifier(fused).squeeze(-1), weights
