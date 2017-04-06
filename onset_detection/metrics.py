import numpy as np


def onset_metric(y, y_actual_onset_only, y_predicted, n_tolerance_frames_plus_minus):
    """Assumes y is grouped by wav file, sorted by time asc.

    n_tolerance_frames_plus_minus: n_tolerance_frames_plus_minus before and n_tolerance_frames_plus_minus after
    the frame labelled as onset will also count as true positive.
    """

    actual_onset_indices = np.where(y_actual_onset_only == 1)[0]
    # Make sure all actual onset indices are indeed labelled as onsets in the ground truth y used for training.
    # Otherwise something is off.
    assert len(actual_onset_indices) == np.sum(y[actual_onset_indices] == 1)

    tp = 0
    fn = 0
    tolerated_onset_indices = set()
    for actual_onset_index in actual_onset_indices:
        is_tp = False
        start = max(0, actual_onset_index - n_tolerance_frames_plus_minus)
        end = min(len(y), actual_onset_index + n_tolerance_frames_plus_minus + 1)
        for i in range(start, end):
            tolerated_onset_indices.add(i)
            # If any of the labels equals 1 in the region of frame_tol frames: mark as true positive.
            if y_predicted[i] == 1:
                is_tp = True
        if is_tp:
            tp += 1
        else:
            fn += 1

    # Create index mask which is the inverse of the tolerated onset indices to detect false positives.
    tolerated_onset_indices_array = np.array(list(tolerated_onset_indices), dtype=np.int32)
    fp_mask = np.ones(len(y), np.bool)
    fp_mask[tolerated_onset_indices_array] = 0
    fp = np.sum(y_predicted[fp_mask] == 1)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    result_string = 'TP=' + str(tp) + ', FN=' + str(fn) + ', FP=' + str(fp) + '\n'
    result_string += 'precision=' + str(precision) + ', recall=' + str(recall) + ', F1=' + str(f1) + '\n'

    return result_string
