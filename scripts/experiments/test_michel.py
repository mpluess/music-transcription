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


import numpy as np
a = np.array([[0.1, 0.6], [0.54, 0.44]])
print(a)
print(a > 0.5)
