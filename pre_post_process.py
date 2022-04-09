import cv2
import numpy as np


# several useful functions:
def show_image(image):
    cv2.namedWindow('image', 0)
    cv2.resizeWindow('image', 600, 500)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_image_with_bbox(image, box):

    """
    show_picture: print the picture with window size 600 * 500
    :param image: picture matrix
    :param box: bounding of the objection, containing x_center, y_center, weight and height
    :return: none
    """

    first_x = np.round(box[0] - 0.5 * box[2]).astype(np.int64)
    first_y = np.round(box[1] - 0.5 * box[3]).astype(np.int64)
    last_x = np.round(box[0] + 0.5 * box[2]).astype(np.int64)
    last_y = np.round(box[1] + 0.5 * box[3]).astype(np.int64)
    first_point = first_x, first_y
    last_point = last_x, last_y
    new_picture = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(new_picture, first_point, last_point, (0, 255, 0), 1)  # channels: B G R
    cv2.namedWindow('image', 0)
    cv2.resizeWindow('image', 600, 500)
    cv2.imshow('image', new_picture)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_image_with_bbox(image, box, color):

    """
    picture_with_box: add box in the picture, which contains the object
    :param image: picture matrix
    :param box: bounding of the objection, containing x_center, y_center, weight and height
    :param color: color of the box
    :return: the new picture with the box
    """
    first_x = np.round(box[0] - 0.5 * box[2]).astype(np.int64)
    first_y = np.round(box[1] - 0.5 * box[3]).astype(np.int64)
    last_x = np.round(box[0] + 0.5 * box[2]).astype(np.int64)
    last_y = np.round(box[1] + 0.5 * box[3]).astype(np.int64)
    if first_x < 0:
        first_x = 0
    elif first_x >= 240:
        first_x = 239
    if last_x < 0:
        last_x = 0
    elif last_x >= 240:
        last_x = 239
    if first_y < 0:
        first_y = 0
    elif first_y >= 180:
        first_y = 180
    if last_y < 0:
        last_y = 0
    elif last_y >= 180:
        last_y = 179
    first_point = int(first_x), int(first_y)
    last_point = int(last_x), int(last_y)
    if len(image.shape) == 2:
        new_picture = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        new_picture = image
    cv2.rectangle(new_picture, first_point, last_point, color, 1)  # channels: B G R
    return new_picture


# filling the rectangle picture into a square picture, measure 448 * 448
def image_squaring(image):
    w, h = image.shape  # w = 180, h = 240
    side_length = int((h - w) / 2)  # 30
    filling = np.ones(shape=(side_length, h)) * 127
    result = np.concatenate((filling, image, filling)).astype(np.uint8)
    result = cv2.resize(result, (224, 224))
    return result


# change the label corresponding to the image
def label_transforming(label: np.array):
    result = label.copy()
    result[1] += 30  # the image is filled
    result = result * 224 / 240
    return result


# recover the label to normal piece, which enables us to draw the picture
def label_recovering(label):
    result = label.copy() * 240 / 224
    result[1] -= 30
    return result


def make_video(image_set, file_path, label_predicted, label_ground_truth=None):
    # video writer
    frame_size = (240, 180)
    ground_truth_color = (0, 255, 0)  # the box of ground truth is green
    predicted_color = (0, 0, 255)  # the box of prediction is red
    video_writer = cv2.VideoWriter(file_path, -1, 20, frame_size, True)
    for i in range(len(image_set)):
        if label_ground_truth is None:
            temp_frame = get_image_with_bbox(image_set[i], label_predicted[i], predicted_color)
        else:
            temp_frame = get_image_with_bbox(image_set[i], label_ground_truth[i], ground_truth_color)
            temp_frame = get_image_with_bbox(temp_frame, label_predicted[i], predicted_color)
        video_writer.write(temp_frame)
    video_writer.release()


# used to compute the intersection over union
def compute_IoU(bbox_pre, bbox_ground_truth):
    bbox_pre_x_min = np.round(bbox_pre[0] - 0.5 * bbox_pre[2]).astype(np.int64)
    bbox_pre_x_max = np.round(bbox_pre[0] + 0.5 * bbox_pre[2]).astype(np.int64)
    bbox_pre_y_min = np.round(bbox_pre[1] - 0.5 * bbox_pre[3]).astype(np.int64)
    bbox_pre_y_max = np.round(bbox_pre[1] + 0.5 * bbox_pre[3]).astype(np.int64)

    bbox_ground_truth_x_min = np.round(bbox_ground_truth[0] - 0.5 * bbox_ground_truth[2]).astype(np.int64)
    bbox_ground_truth_x_max = np.round(bbox_ground_truth[0] + 0.5 * bbox_ground_truth[2]).astype(np.int64)
    bbox_ground_truth_y_min = np.round(bbox_ground_truth[1] - 0.5 * bbox_ground_truth[3]).astype(np.int64)
    bbox_ground_truth_y_max = np.round(bbox_ground_truth[1] + 0.5 * bbox_ground_truth[3]).astype(np.int64)

    inter_x_min = max(bbox_pre_x_min, bbox_ground_truth_x_min)
    inter_x_max = min(bbox_pre_x_max, bbox_ground_truth_x_max)
    inter_x = inter_x_max - inter_x_min

    inter_y_min = max(bbox_pre_y_min, bbox_ground_truth_y_min)
    inter_y_max = min(bbox_pre_y_max, bbox_ground_truth_y_max)
    inter_y = inter_y_max - inter_y_min

    inter = 0 if inter_x < 0 or inter_y < 0 else inter_x * inter_y
    union = bbox_pre[2] * bbox_pre[3] + bbox_ground_truth[2] * bbox_ground_truth[3] - inter
    return inter / union
