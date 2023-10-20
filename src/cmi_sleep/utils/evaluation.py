from typing import Tuple

import numpy as np
import numpy.typing as npt
from scipy.signal import savgol_filter, find_peaks
from scipy.ndimage import binary_closing


def filter_predictions(
    y_proba: npt.ArrayLike, window_size: int, thres: float = 0.5
) -> npt.ArrayLike:
    """Filter predictions by first applying a smoothing filter and then doing a binary
    thresholding followed by a closing operation to close gaps in the prediction
    smaller than 'window_size'"""
    window_size = min(len(y_proba), window_size)
    y_proba_smooth = savgol_filter(y_proba, window_length=window_size, polyorder=2)
    y_pred = y_proba_smooth > thres
    if (window_size % 2) == 0:
        window_size += 1
    y_pred = binary_closing(y_pred, structure=np.ones(window_size)) * 1
    return y_pred


def event_indices_from_predictions(
    y_pred: npt.ArrayLike, min_event_size: int
) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
    """Get event indicies from the predictions throgh the 'find_peaks' scipy function.
    Events smaller than 'min_event_size' are discarded"""
    peaks, plateaus = find_peaks(y_pred, plateau_size=min_event_size)
    onset_idxs = plateaus["left_edges"]
    wakeup_idxs = plateaus["right_edges"]
    return onset_idxs, wakeup_idxs


def score_events(
    y_proba: npt.ArrayLike, onset_idxs: npt.ArrayLike, wakeup_idxs: npt.ArrayLike
) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
    """Score onset and wakeup events by taking a linearly decaying average of all
    probabilities within the sleeping period. Probabilities closer to the predicted
    index have a higher weight"""
    onset_scores = np.zeros(len(onset_idxs))
    wakeup_scores = np.zeros(len(wakeup_idxs))
    for event_idx, (start_idx, end_idx) in enumerate(zip(onset_idxs, wakeup_idxs)):
        event_y_proba = y_proba[start_idx:end_idx]
        lin_weight = np.linspace(0, 1, num=len(event_y_proba))
        onset_scores[event_idx] = np.sum(event_y_proba * lin_weight[::-1]) / np.sum(
            lin_weight
        )
        wakeup_scores[event_idx] = np.sum(event_y_proba * lin_weight) / np.sum(
            lin_weight
        )
    return onset_scores, wakeup_scores
