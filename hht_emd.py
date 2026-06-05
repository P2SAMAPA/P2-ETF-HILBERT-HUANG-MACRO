import numpy as np
from scipy.signal import hilbert

def emd_sift(signal, max_imfs=5, max_sift_iter=10, tol=0.01):
    """
    Empirical Mode Decomposition (EMD) using sifting.
    Returns array (n_imfs, n_samples).
    """
    n = len(signal)
    if n < 3:
        return np.array([signal])
    imfs = []
    residual = signal.copy()
    for _ in range(max_imfs):
        h = residual.copy()
        for _ in range(max_sift_iter):
            # Find upper and lower envelopes via cubic spline interpolation of local maxima/minima
            maxima_idx = np.where((h[1:-1] > h[:-2]) & (h[1:-1] > h[2:]))[0] + 1
            minima_idx = np.where((h[1:-1] < h[:-2]) & (h[1:-1] < h[2:]))[0] + 1
            if len(maxima_idx) < 2 or len(minima_idx) < 2:
                break
            # Cubic spline interpolation
            from scipy.interpolate import CubicSpline
            t = np.arange(n)
            upper = CubicSpline(maxima_idx, h[maxima_idx], bc_type='natural')(t)
            lower = CubicSpline(minima_idx, h[minima_idx], bc_type='natural')(t)
            # Mean envelope
            m = (upper + lower) / 2.0
            h_new = h - m
            # Check stoppage
            if np.sum(np.abs(h - h_new)) < tol * np.sum(np.abs(h)):
                h = h_new
                break
            h = h_new
        imfs.append(h)
        residual = residual - h
        if np.all(np.abs(residual) < 1e-6):
            break
    return np.array(imfs)

def compute_imfs(series, n_imfs=None):
    if len(series) < 5:
        return np.array([series])
    imfs = emd_sift(series, max_imfs=n_imfs if n_imfs else 5)
    if n_imfs is not None and imfs.shape[0] > n_imfs:
        imfs = imfs[:n_imfs]
    return imfs

def instantaneous_amplitude(imf):
    analytic = hilbert(imf)
    return np.abs(analytic)

def hht_score(returns, macro_value, macro_threshold=None, mode_selection='macro_high_freq', num_imfs=5):
    if len(returns) < 5:
        return 0.0
    imfs = compute_imfs(returns, n_imfs=num_imfs)
    n_imfs = imfs.shape[0]
    if n_imfs == 0:
        return 0.0
    last_amp = []
    for i in range(n_imfs):
        amp = instantaneous_amplitude(imfs[i])
        last_amp.append(amp[-1] if len(amp) > 0 else 0.0)
    if macro_threshold is None:
        macro_threshold = 0.5
    if mode_selection == 'macro_high_freq':
        if macro_value > macro_threshold:
            selected_weights = [1.0 if i < n_imfs/2 else 0.0 for i in range(n_imfs)]
        else:
            selected_weights = [1.0 if i >= n_imfs/2 else 0.0 for i in range(n_imfs)]
    else:
        selected_weights = [1.0/n_imfs] * n_imfs
    score = np.sum([w * a for w, a in zip(selected_weights, last_amp)])
    return float(score)

def hht_macro_score(returns, macro_series, mode_selection='macro_high_freq', num_imfs=5):
    if len(returns) != len(macro_series):
        min_len = min(len(returns), len(macro_series))
        returns = returns[:min_len]
        macro_series = macro_series[:min_len]
    if len(returns) < 5:
        return 0.0
    macro_threshold = np.median(macro_series)
    current_macro = macro_series[-1]
    score = hht_score(returns, current_macro, macro_threshold, mode_selection, num_imfs)
    return score
