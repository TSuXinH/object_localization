import h5py
import numpy as np

# data pre-procession
data_root = './dataset'
train_data = h5py.File(data_root + '/train_aps_x.mat', 'r')['train_aps_x'][:]
train_bbox = h5py.File(data_root + '/train_aps_y.mat', 'r')['train_aps_y'][:]
test_data = h5py.File(data_root + '/test_aps_x.mat', 'r')['test_aps_x'][:]
test_bbox = h5py.File(data_root + '/test_aps_y.mat', 'r')['test_aps_y'][:]

x_train = []
for i in range(train_data.shape[2]):
    x_train.append(train_data[:, :, i].T)
x_test = []
for i in range(test_data.shape[2]):
    x_test.append(test_data[:, :, i].T)
# this part is used for drawing the bounding box, the pixels are integer
all_train = np.array(x_train)
all_label = train_bbox.T[:, : 4].astype('float')
original_train = np.concatenate((np.array(x_train)[: 600], np.array(x_train)[700:]), axis=0)
original_test = np.array(x_test)
original_train_bbox = train_bbox.T[:, : 4].astype('float')
original_train_bbox = np.concatenate((original_train_bbox[:600], original_train_bbox[700:]), axis=0)
original_test_bbox = test_bbox.T[:, : 4].astype('float')

