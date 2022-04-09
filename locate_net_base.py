import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from base import *
from locate_net import *
from locate_pre_processing import *

modified_train_data, modified_train_label = dataset_pre_processing(original_train, original_train_bbox)
modified_test_data, modified_test_label = dataset_pre_processing(original_test, original_test_bbox)

# this part is used for training and testing
x_train_ = torch.true_divide(torch.unsqueeze(torch.tensor(modified_train_data), dim=1), 255)
x_test_ = torch.true_divide(torch.unsqueeze(torch.tensor(modified_test_data), dim=1), 255)
y_train_ = torch.tensor(modified_train_label).float()
y_test_ = torch.tensor(modified_test_label).float()

train_loader = DataLoader(dataset=TensorDataset(x_train_, y_train_), shuffle=True, batch_size=16)
demo_train_loader = DataLoader(dataset=TensorDataset(x_train_, y_train_), shuffle=False, batch_size=16)
test_loader = DataLoader(dataset=TensorDataset(x_test_, y_test_), shuffle=False, batch_size=16)

all_modified_train, all_modified_label = dataset_pre_processing(all_train, all_label)
all_t = torch.true_divide(torch.unsqueeze(torch.tensor(all_modified_train), dim=1), 255)
all_l = torch.tensor(all_modified_label).float()
all_train_loader = DataLoader(dataset=TensorDataset(all_t, all_l), shuffle=False, batch_size=16)

_lr = 1e-3
_wd = 1e-4
epoch_num = 1
net = locate_net()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=_lr, weight_decay=_wd)

# train part
for i in range(epoch_num):
    total_loss = 0
    for idx, (data, label) in enumerate(train_loader):
        data, label = data, label
        output = net(data)
        loss = criterion(output, label)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('epoch {} done, loss: {}'.format(i + 1, total_loss / len(train_loader.dataset)))

y_predicted = np.array([])
with torch.no_grad():
    for idx, (data, label) in enumerate(train_loader):
        output = net(data)
        temp = output.numpy()
        temp = reconstruct_label(temp)
        if len(y_predicted) == 0:
            y_predicted = temp
        else:
            y_predicted = np.concatenate((y_predicted, temp), axis=0)

# get the result from google colaboratory
# show train result
loc_all_train = np.load('./train_v2.npy')
make_video(original_train, './loc_all.mp4', loc_all_train, original_train_bbox)

# show test result
loc_test = np.load('./loc_test.npy')
make_video(original_test, './loc_test.mp4', loc_test)

