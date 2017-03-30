import numpy as np


def onset_metric(y, y_start_only, y_predicted, frame_tol=5):
    """Assumes y is grouped by wav file, sorted by time asc."""

    onset_start_indices = np.where(y_start_only == 1)[0]
    # Make sure all onset start indices are indeed labelled as onsets in the actual ground truth y. Otherwise something is off.
    assert len(onset_start_indices) == np.sum(y[onset_start_indices] == 1)

    tp = 0
    fn = 0
    onset_indices = set()
    for start_index in onset_start_indices:
        is_tp = False
        for i in range(start_index, start_index + frame_tol):
            onset_indices.add(i)
            # If any of the labels equals 1 in the region of frame_tol frames: mark as true positive.
            if y_predicted[i] == 1:
                is_tp = True
        if is_tp:
            tp += 1
        else:
            fn += 1

    # Create index mask which is the inverse of the tolerated onset indices to detect false positives.
    onset_indices_array = np.array(list(onset_indices), dtype=np.int32)
    fp_mask = np.ones(len(y), np.bool)
    fp_mask[onset_indices_array] = 0
    fp = np.sum(y_predicted[fp_mask] == 1)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    result_string = 'TP=' + str(tp) + ', FN=' + str(fn) + ', FP=' + str(fp) + '\n'
    result_string += 'precision=' + str(precision) + ', recall=' + str(recall) + ', F1=' + str(f1) + '\n'

    return result_string
