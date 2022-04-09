from pre_post_process import *

import numpy as np


# transform the data and label for training
def dataset_pre_processing(data, label):
    result_data = []
    result_label = np.zeros(shape=(len(label), 7, 7, 10))  # using 2 bounding boxes for predicting
    for i in range(len(label)):
        temp_data = image_squaring(data[i])
        result_data.append(temp_data)
        grid_x = int(label[i][0] // 32)
        grid_y = int(label[i][1] // 32)
        t_x = (label[i][0] - grid_x * 32) / 32
        t_y = (label[i][1] - grid_y * 32) / 32
        t_w = np.log(label[i][2] / 224)
        t_h = np.log(label[i][3] / 224)
        result_label[i][grid_x][grid_y] = np.array([t_x, t_y, t_w, t_h, 1, t_x, t_y, t_w, t_h, 1])
    result_data = np.array(result_data)
    return result_data, result_label


# for testing the code
def show_label(single_label):
    for i in range(7):
        for j in range(7):
            print(single_label[i][j])


# after getting the label, reconstruct the image, label (7 * 7 * 5)
def reconstruct_label(label):
    result_label = np.zeros(shape=(len(label), 4))
    for i in range(len(label)):
        max_confidence = 0
        max_coor = []
        for j in range(7):
            for k in range(7):
                if max_confidence < label[i][j][k][4]:
                    max_confidence = label[i][j][k][4]
                    max_coor = [j, k]
        result_label[i][0] = (label[i][max_coor[0]][max_coor[1]][0] + max_coor[0]) * 32
        result_label[i][1] = (label[i][max_coor[0]][max_coor[1]][1] + max_coor[1]) * 32
        result_label[i][2] = np.exp(label[i][max_coor[0]][max_coor[1]][2]) * 224
        result_label[i][3] = np.exp(label[i][max_coor[0]][max_coor[1]][3]) * 224
    return result_label


def make_video_YOLO(image_set, file_path, threshold, label_predicted, label_ground_truth=None):
    # video writer
    frame_size = (240, 180)
    ground_truth_color = (0, 255, 0)  # the box of ground truth is green
    predicted_color = (0, 0, 255)  # the box of prediction is red
    video_writer = cv2.VideoWriter(file_path, -1, 20, frame_size, True)
    for i in range(len(image_set)):
        # if confidence is less than 0.9, we don't show the predicted bounding box
        if label_ground_truth is None:
            if label_predicted[i][4] < threshold:
                temp_frame = image_set[i]
            else:
                temp_frame = get_image_with_bbox(image_set[i], label_predicted[i], predicted_color)
        else:
            if label_predicted[i][4] < threshold:
                temp_frame = get_image_with_bbox(image_set[i],
                                                 label_ground_truth[i], ground_truth_color)
            else:
                temp_frame = get_image_with_bbox(image_set[i],
                                                 label_ground_truth[i], ground_truth_color)
                temp_frame = get_image_with_bbox(temp_frame, label_predicted[i], predicted_color)
        video_writer.write(temp_frame)
    video_writer.release()
