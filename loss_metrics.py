from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Optional

try:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
    from sklearn.preprocessing import label_binarize
    _SK_AVAILABLE = True
except Exception:
    _SK_AVAILABLE = False


def make_ce_loss(class_weights: Optional[torch.Tensor] = None):
    if class_weights is not None:
        return torch.nn.CrossEntropyLoss(weight=class_weights)
    return torch.nn.CrossEntropyLoss()


@torch.no_grad()
def _safe_softmax(logits: torch.Tensor) -> torch.Tensor:
    # safer for fp16 logits
    x = logits.float()
    m = x.max(dim=1, keepdim=True).values
    ex = torch.exp(x - m)
    probs = ex / ex.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return probs.to(dtype=logits.dtype)


@torch.no_grad()
def compute_metrics(
    logits: torch.Tensor,   # (N,C)
    targets: torch.Tensor,  # (N,)
    average: str = "macro",
) -> Dict[str, float]:

    probs = _safe_softmax(logits)
    preds = torch.argmax(probs, dim=1)

    y_true = targets.detach().cpu().numpy()
    y_pred = preds.detach().cpu().numpy()
    y_proba = probs.detach().cpu().numpy()

    out: Dict[str, float] = {}

    if _SK_AVAILABLE:
        out["accuracy"] = float(accuracy_score(y_true, y_pred))
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=average, zero_division=0
        )
        out["precision"] = float(p)
        out["recall"]    = float(r)
        out["f1"]        = float(f1)
    else:
        acc = float((preds == targets).float().mean().item())
        out["accuracy"] = acc

        num_classes = logits.shape[1]
        f1_list = []
        for c in range(num_classes):
            tp = ((preds == c) & (targets == c)).sum().item()
            fp = ((preds == c) & (targets != c)).sum().item()
            fn = ((preds != c) & (targets == c)).sum().item()
            precision = tp / (tp + fp + 1e-8)
            recall    = tp / (tp + fn + 1e-8)
            f1_c = 2 * precision * recall / (precision + recall + 1e-8)
            f1_list.append(f1_c)
        out["precision"] = float("nan")
        out["recall"]    = float("nan")
        out["f1"]        = float(np.mean(f1_list))

    if not _SK_AVAILABLE:
        out["auc_macro_ovr"]    = float("nan")
        out["auc_weighted_ovr"] = float("nan")
        out["auc_micro_ovr"]    = float("nan")
        out["auc_ovr"]          = float("nan")
        return out

    classes_present = np.unique(y_true).astype(int).tolist()
    C = y_proba.shape[1]

    # per-class one-vs-rest AUC (only when both pos/neg exist)
    per_class_auc = []
    per_class_support = []
    valid_auc_classes = []

    for c in range(C):
        y_bin = (y_true == c).astype(np.uint8)
        pos = int(y_bin.sum())
        neg = int(len(y_bin) - pos)
        if pos > 0 and neg > 0:
            try:
                auc_c = float(roc_auc_score(y_bin, y_proba[:, c]))
                per_class_auc.append(auc_c)
                per_class_support.append(int(pos))
                valid_auc_classes.append(c)
            except Exception:
                pass

    if len(per_class_auc) == 0:
        out["auc_macro_ovr"]    = float("nan")
        out["auc_weighted_ovr"] = float("nan")
        out["auc_micro_ovr"]    = float("nan")
        out["auc_ovr"]          = float("nan")
        return out

    out["auc_macro_ovr"] = float(np.mean(per_class_auc))
    out["auc_weighted_ovr"] = float(np.average(per_class_auc, weights=np.array(per_class_support)))
    out["auc_ovr"] = out["auc_macro_ovr"]

    if len(valid_auc_classes) >= 2:
        mask = np.isin(y_true, valid_auc_classes)
        y_true_f = y_true[mask]
        y_proba_f = y_proba[mask][:, valid_auc_classes]

        y_bin = label_binarize(y_true_f, classes=valid_auc_classes)
        try:
            out["auc_micro_ovr"] = float(roc_auc_score(y_bin, y_proba_f, average="micro"))
        except Exception:
            out["auc_micro_ovr"] = float("nan")
    else:
        out["auc_micro_ovr"] = float("nan")

    return out
