# import numpy as np
#
# n_frames_after_cutoff_per_file = [10, 15, 5]
# print(n_frames_after_cutoff_per_file)
# # [10 15 5] -> [5 10 15]
# start_index_per_file = np.roll(n_frames_after_cutoff_per_file, 1)
# print(start_index_per_file)
# # [5 10 15] -> [0 10 15]
# start_index_per_file[0] = 0
# print(start_index_per_file)
# # [0 10 15] -> [0 10 25]
# start_index_per_file = np.cumsum(start_index_per_file)
# print(start_index_per_file)


# print(float(0.0000001))


# from sklearn.preprocessing import MultiLabelBinarizer
#
# label_binarizer = MultiLabelBinarizer(classes=range(40, 89))
# label_binarizer.fit(None)
# print(label_binarizer.transform([{40, 42, 43}, {81, 83, 88}, {50}]))


# import numpy as np
# a = np.array([[0.1, 0.6], [0.54, 0.44]])
# print(a)
# print(a > 0.5)

import os
import pickle
from zipfile import ZipFile

config = {
    'onset_group_threshold_seconds': 0.05
}
config_file = r'D:\Users\Michel\Documents\FH\module\8_IP6\git\models\onset_detection\config.pickle'
with open(config_file, 'wb') as f:
    pickle.dump(config, f)

to_zip = [
    config_file,
    r'D:\Users\Michel\Documents\FH\module\8_IP6\git\models\onset_detection\feature_extractor.pickle',
    r'D:\Users\Michel\Documents\FH\module\8_IP6\git\models\onset_detection\model.json',
    r'D:\Users\Michel\Documents\FH\module\8_IP6\git\models\onset_detection\weights.hdf5',
]
with ZipFile(r'D:\Users\Michel\Documents\FH\module\8_IP6\git\models\onset_detection\20170627-3-channels_ds1-4_80-perc_adjusted-labels_with_config_thresh-0.05.zip', 'w') as zip_file:
    for path_to_file in to_zip:
        zip_file.write(path_to_file, arcname=os.path.basename(path_to_file))

# a = 22.239999999999998
# b = 22.27
# c = 22.289999999999999
# epsilon=1e-6
# onset_group_threshold_seconds = 0.03
#
# print(b - a)
# print(c - b)
# print(onset_group_threshold_seconds - epsilon)
# print(onset_group_threshold_seconds + epsilon)

# print(5 / 3)
# print(round(5 / 3, 3))
