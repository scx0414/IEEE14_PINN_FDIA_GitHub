"""A system-level target is fitted to one maximum node residual per timestamp."""

import numpy as np
from scipy.stats import weibull_min


def fit_weibull(node_scores, alpha=0.05):
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in(0,1)")
    v = np.asarray(node_scores, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] != 14:
        raise ValueError("Expected[N,14] node scores")
    s = v.max(1)
    if not np.isfinite(s).all() or np.any(s <= 0):
        raise ValueError("Weibull fitting requires finite positive system scores")
    k, _, scale = weibull_min.fit(s, floc=0)
    threshold = float(scale * (-np.log(alpha)) ** (1 / k))
    return dict(
        shape=float(k),
        scale=float(scale),
        threshold=threshold,
        alpha=alpha,
        calibration_points=len(s),
        calibration_system_fpr=float(np.mean(s > threshold)),
    )


def classify(node_scores, threshold):
    return np.asarray(node_scores).max(1) > threshold


def confusion(alarm, truth):
    a, y = np.asarray(alarm, dtype=bool), np.asarray(truth, dtype=bool)
    if a.shape != y.shape or a.ndim != 1 or not len(a):
        raise ValueError("Alarm and label must have matching nonempty1D shapes")
    tp, fn, fp, tn = [int(v.sum()) for v in (a & y, ~a & y, a & ~y, ~a & ~y)]
    recall, fpr = tp / (tp + fn) if tp + fn else 0.0, fp / (fp + tn) if fp + tn else 0.0
    return dict(
        tp=tp,
        fn=fn,
        fp=fp,
        tn=tn,
        recall=recall,
        fpr=fpr,
        precision=tp / (tp + fp) if tp + fp else 0.0,
        f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        accuracy=(tp + tn) / len(a),
        balanced_accuracy=(recall + 1 - fpr) / 2,
    )


def event_metrics(alarm, event_ids, sampling_minutes=5):
    results = []
    for event in np.unique(event_ids):
        if event == 0:
            continue
        indices = np.flatnonzero(event_ids == event)
        hits = np.flatnonzero(np.asarray(alarm)[indices])
        results.append(
            dict(
                event_id=int(event),
                detected=bool(len(hits)),
                delay_minutes=int(hits[0] * sampling_minutes) if len(hits) else None,
            )
        )
    return results
