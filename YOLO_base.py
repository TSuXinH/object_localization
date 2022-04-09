from base import *
from YOLO_net import *
from YOLO_loss import *
from YOLO_pre_processing import *

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import tensorboardX
from tensorboardX import SummaryWriter

# modify the data for training and testing
modified_train_data, modified_train_label = dataset_pre_processing(original_train, original_train_bbox)
modified_test_data, modified_test_label = dataset_pre_processing(original_test, original_test_bbox)

# this part is used for train and test loader
x_train_ = torch.true_divide(torch.unsqueeze(torch.tensor(modified_train_data), dim=1), 255)
x_test_ = torch.true_divide(torch.unsqueeze(torch.tensor(modified_test_data), dim=1), 255)
y_train_ = torch.tensor(modified_train_label).float()
y_test_ = torch.tensor(modified_test_label).float()

train_loader = DataLoader(dataset=TensorDataset(x_train_, y_train_), shuffle=True, batch_size=64)
demo_train_loader = DataLoader(dataset=TensorDataset(x_train_, y_train_), shuffle=False, batch_size=64)
test_loader = DataLoader(dataset=TensorDataset(x_test_, y_test_), shuffle=False, batch_size=64)

all_modified_train, all_modified_label = dataset_pre_processing(all_train, all_label)
all_t = torch.true_divide(torch.unsqueeze(torch.tensor(all_modified_train), dim=1), 255)
all_l = torch.tensor(all_modified_label).float()
all_train_loader = DataLoader(dataset=TensorDataset(all_t, all_l), shuffle=False, batch_size=64)

# construct net, criterion and optimizer
_lr = 1e-4
_wd = 0
_mom = 0.9
epoch_num = 1
YOLO_net = YOLO_v1()
criterion = YOLO_loss()
optimizer = torch.optim.SGD(params=YOLO_net.parameters(), lr=_lr, momentum=_mom, weight_decay=0)

# train part
YOLO_net.train()
for i in range(epoch_num):
    total_loss = 0
    for idx, (data, label) in enumerate(train_loader):
        output = YOLO_net(data)
        loss = criterion(output, label)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('epoch {} done, loss: {}'.format(i + 1, total_loss))

# # get the parameters of the net from google colaboratory
# torch.save(YOLO_net, './net.pkl')
# YOLO_net = torch.load('./net.pkl')

y_predicted = np.array([])
YOLO_net.eval()
with torch.no_grad():
    for idx, (data, label) in enumerate(train_loader):
        output = YOLO_net(data)
        temp = output.numpy()
        temp = reconstruct_label(temp)
        if len(y_predicted) == 0:
            y_predicted = temp
        else:
            y_predicted = np.concatenate((y_predicted, temp), axis=0)
final_label = []
for i in range(len(y_predicted)):
    final_label.append(label_recovering(y_predicted[i]))
final_label = np.array(final_label)

# get the data which has been predicted from google colaboratory and make the video
final_label = np.load('./y_pre_all.npy')
# make a video
make_video_YOLO(all_train, './all_train.mp4', 0.85, final_label, all_label)

test_label = np.load('./test_pre.npy')
make_video_YOLO(original_test, './test.mp4', 0.85, test_label)

x = torch.rand(64, 1, 224, 224)
model = YOLO_v1()
with SummaryWriter(comment='YOLO') as w:
    w.add_graph(model, x)
