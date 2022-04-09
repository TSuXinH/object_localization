import torch
import torch.nn as nn


class YOLO_loss(nn.Module):
    def __init__(self):
        super(YOLO_loss, self).__init__()

    def forward(self, label_pre, label_ground_truth):
        loss = torch.Tensor([0.])
        for i in range(len(label_pre)):
            for j in range(7):
                for k in range(7):
                    if label_ground_truth[i][j][k][4] == 1:
                        loss += 5 * torch.pow((label_pre[i][j][k][0] - label_ground_truth[i][j][k][0]), 2)
                        loss += 5 * torch.pow((label_pre[i][j][k][1] - label_ground_truth[i][j][k][1]), 2)
                        loss += 5 * torch.pow((label_pre[i][j][k][2] - label_ground_truth[i][j][k][2]), 2)
                        loss += 5 * torch.pow((label_pre[i][j][k][3] - label_ground_truth[i][j][k][3]), 2)
                        loss += 10 * torch.pow((label_pre[i][j][k][4] - label_ground_truth[i][j][k][4]), 2)
                    else:
                        loss += 0.5 * torch.pow((label_pre[i][j][k][4] - label_ground_truth[i][j][k][4]), 2)
        return loss
