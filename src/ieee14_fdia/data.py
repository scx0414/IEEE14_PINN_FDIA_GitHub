import numpy as np


def fit_scalers(z, x, train_end):
    if not 1 < train_end <= len(z):
        raise ValueError("Invalid training endpoint")
    return dict(
        input_mean=z[:train_end].mean(0),
        input_scale=np.maximum(z[:train_end].std(0), 1e-6),
        state_mean=x[:train_end].mean(0),
        state_scale=np.maximum(x[:train_end].std(0), 1e-3),
    )


def make_windows(z, window, past=None):
    """Current-input windows; optional past contains only strictly previous observations."""
    z = np.asarray(z)
    if window < 1:
        raise ValueError("window must be positive")
    if past is not None:
        if len(past) < window - 1:
            raise ValueError("Not enough past context")
        if window > 1:
            z = np.concatenate([np.asarray(past)[-window + 1 :], z])
    if len(z) < window:
        raise ValueError("Sequence shorter than window")
    return np.lib.stride_tricks.sliding_window_view(z, window, axis=0).transpose(
        0, 2, 1
    )


def normalize(windows, scalers):
    return ((windows - scalers["input_mean"]) / scalers["input_scale"]).astype(
        np.float32
    )
