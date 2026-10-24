"""Observed metrics only; no hard-coded paper scores."""
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def binary_metrics(labels, probabilities, threshold=0.5):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if not len(labels) or labels.shape != probabilities.shape or not np.isfinite(probabilities).all():
        raise ValueError("Invalid metric inputs")
    predicted = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    two_classes = len(np.unique(labels)) == 2
    return dict(n=len(labels), threshold=threshold, accuracy=float(accuracy_score(labels, predicted)),
                precision=float(precision_score(labels, predicted, zero_division=0)),
                sensitivity=float(recall_score(labels, predicted, zero_division=0)),
                specificity=float(tn / (tn + fp)) if tn + fp else None,
                f1=float(f1_score(labels, predicted, zero_division=0)),
                roc_auc=float(roc_auc_score(labels, probabilities)) if two_classes else None,
                average_precision=float(average_precision_score(labels, probabilities)) if two_classes else None,
                brier_score=float(brier_score_loss(labels, probabilities)),
                confusion_matrix=dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp)))
