import numpy as np
from scipy.signal import hilbert
from EMD import EMD
import warnings
warnings.filterwarnings("ignore")

def compute_imfs(series, n_imfs=None):
    """
    Decompose signal into Intrinsic Mode Functions (IMFs) using EMD.
    Returns array (n_imfs, n_samples).
    """
    if len(series) < 10:
        return np.array([series])
    emd = EMD()
    imfs = emd(series)
    if n_imfs is not None and imfs.shape[0] > n_imfs:
        imfs = imfs[:n_imfs]
    return imfs

def instantaneous_amplitude(imf):
    """Hilbert transform to get instantaneous amplitude envelope."""
    analytic = hilbert(imf)
    return np.abs(analytic)

def hht_score(returns, macro_value, macro_threshold=None, mode_selection='macro_high_freq'):
    """
    Decompose returns into IMFs, compute last point's amplitude for each IMF,
    then select IMFs based on macro regime.
    Returns a weighted sum (score).
    """
    if len(returns) < 10:
        return 0.0
    imfs = compute_imfs(returns, n_imfs=config.NUM_IMFS)
    n_imfs = imfs.shape[0]
    # Compute instantaneous amplitude for each IMF at the last point
    last_amp = []
    for i in range(n_imfs):
        amp = instantaneous_amplitude(imfs[i])
        last_amp.append(amp[-1])
    # Determine which IMFs are "high frequency" (earlier IMFs) vs low frequency
    # High-frequency IMFs have index 0,1,...; low-frequency have higher indices
    # Macro threshold: if macro_value > macro_threshold (e.g., median), select high-freq IMFs
    if macro_threshold is None:
        macro_threshold = np.median(macro_value) if isinstance(macro_value, np.ndarray) else 0.5
    if mode_selection == 'macro_high_freq':
        # High macro -> select high-freq IMFs (low indices), else low-freq IMFs
        if macro_value > macro_threshold:
            selected_weights = [1.0 if i < n_imfs/2 else 0.0 for i in range(n_imfs)]
        else:
            selected_weights = [1.0 if i >= n_imfs/2 else 0.0 for i in range(n_imfs)]
    else:
        # simple: all IMFs equally weighted
        selected_weights = [1.0/n_imfs] * n_imfs
    # Weighted sum of last amplitudes
    score = np.sum([w * a for w, a in zip(selected_weights, last_amp)])
    return float(score)

def hht_macro_score(returns, macro_series, mode_selection='macro_high_freq'):
    """
    For a given window, compute macro threshold (median of macro over window)
    and then apply hht_score using the last macro value.
    """
    if len(returns) != len(macro_series):
        min_len = min(len(returns), len(macro_series))
        returns = returns[:min_len]
        macro_series = macro_series[:min_len]
    if len(returns) < 10:
        return 0.0
    macro_threshold = np.median(macro_series)
    current_macro = macro_series[-1]
    score = hht_score(returns, current_macro, macro_threshold, mode_selection)
    return score
