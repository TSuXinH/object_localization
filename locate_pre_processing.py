from copy import deepcopy

from pre_post_process import *


# transform the data and label for training
def dataset_pre_processing(data, label):
    result_data = []
    result_label = []
    for i in range(len(label)):
        temp_data = image_squaring(data[i])
        result_data.append(temp_data)
        temp_label = label_transforming(label[i])
        temp_label /= 224
        result_label.append(temp_label)
    result_data = np.array(result_data)
    result_label = np.array(result_label)
    return result_data, result_label


# after getting the label, reconstruct the image, label (7 * 7 * 5)
def reconstruct_label(label):
    result_label = []
    for i in range(len(label)):
        temp = deepcopy(label[i]) * 224
        temp = label_recovering(temp)
        result_label.append(temp)
    result_label = np.array(result_label)
    return result_label
