import random
import os
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torchvision import transforms
from models.model import CNNModel
import numpy as np
from test_script import test_
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torchsummary import summary
import warnings
import torch
import torchvision.transforms as T
import torchvision

device_gpu = 0

warnings.filterwarnings("ignore")
source_dataset_name = 'source'
target_dataset_name = 'target'

model_root = os.path.join(r"D:\UDA-Thermal\UDA_thermal\models", 'models')
cuda = True
cudnn.benchmark = True
lr = 1e-3
batch_size = 32
image_size = 224
n_epoch = 200

manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)

# Load data
img_rgb_resized = np.load(r"D:\UDA-Thermal\UDA_thermal\archive\X_rgb_224.npy")
labels_rgb = np.load(r"D:\UDA-Thermal\UDA_thermal\archive\Y_rgb_224.npy")
img_th_rot = np.load(r"D:\UDA-Thermal\UDA_thermal\archive\X_th_224-001.npy")
labels_th = np.load(r"D:\UDA-Thermal\UDA_thermal\archive\Y_th_224.npy")

img_rgb_train, img_rgb_test, labels_rgb_train, labels_rgb_test = train_test_split(
    img_rgb_resized, labels_rgb, test_size=0.1, random_state=42, stratify=labels_rgb)

img_th_train, img_th_test, labels_th_train, labels_th_test = train_test_split(
    img_th_rot, labels_th, test_size=0.6, random_state=42, stratify=labels_th)

train = torch.utils.data.TensorDataset(torch.from_numpy(img_rgb_train), torch.from_numpy(labels_rgb_train).squeeze(-1))
test = torch.utils.data.TensorDataset(torch.from_numpy(img_rgb_test), torch.from_numpy(labels_rgb_test).squeeze(-1))
train_dataloader_source = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
test_dataloader_source = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)

train_th = torch.utils.data.TensorDataset(torch.from_numpy(img_th_train), torch.from_numpy(labels_th_train).squeeze(-1))
test_th = torch.utils.data.TensorDataset(torch.from_numpy(img_th_test), torch.from_numpy(labels_th_test).squeeze(-1))
train_dataloader_target = torch.utils.data.DataLoader(train_th, batch_size=batch_size, shuffle=True)
test_dataloader_target = torch.utils.data.DataLoader(test_th, batch_size=batch_size, shuffle=True)

dataloader_source = train_dataloader_source
dataloader_target = train_dataloader_target

my_net = CNNModel()

# Setup optimizer
optimizer = optim.Adam(my_net.parameters(), lr=lr)

loss_class = torch.nn.NLLLoss()
loss_domain = torch.nn.NLLLoss()

if cuda:
    my_net = my_net.cuda(device_gpu)
    loss_class = loss_class.cuda(device_gpu)
    loss_domain = loss_domain.cuda(device_gpu)

# Training
src_err = []
src_d_err = []
tgt_d_err = []

source_accu = []
target_accu = []

for epoch in range(n_epoch):

    len_dataloader = min(len(dataloader_source), len(dataloader_target))
    data_source_iter = iter(dataloader_source)
    data_target_iter = iter(dataloader_target)

    i = 0
    while i < len_dataloader:

        p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # Training model using source data
        data_source = next(data_source_iter)
        s_img, s_label = data_source

        my_net.zero_grad()
        batch_size = len(s_label)

        input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
        class_label = torch.LongTensor(batch_size)
        domain_label = torch.zeros(batch_size).long()

        if cuda:
            s_img = s_img.cuda(device_gpu)
            s_label = s_label.cuda(device_gpu)
            input_img = input_img.cuda(device_gpu)
            class_label = class_label.cuda(device_gpu)
            domain_label = domain_label.cuda(device_gpu)

        input_img.resize_as_(s_img).copy_(s_img)
        class_label.resize_as_(s_label).copy_(s_label)

        class_output, domain_output = my_net(input_data=input_img, alpha=alpha)
        err_s_label = loss_class(class_output, class_label)
        err_s_domain = loss_domain(domain_output, domain_label)

        # Training model using target data
        data_target = next(data_target_iter)
        t_img, _ = data_target

        batch_size = len(t_img)

        input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
        domain_label = torch.ones(batch_size).long()

        if cuda:
            t_img = t_img.cuda(device_gpu)
            input_img = input_img.cuda(device_gpu)
            domain_label = domain_label.cuda(device_gpu)

        input_img.resize_as_(t_img).copy_(t_img)

        _, domain_output = my_net(input_data=input_img, alpha=alpha)
        err_t_domain = loss_domain(domain_output, domain_label)
        err = err_t_domain + err_s_domain + err_s_label
        err.backward()
        optimizer.step()

        i += 1 

    src_err.append(err_s_label.cpu().data.numpy())
    src_d_err.append(err_s_domain.cpu().data.numpy())
    tgt_d_err.append(err_t_domain.cpu().data.numpy())
    torch.save(my_net, os.path.join(r'D:\UDA-Thermal\UDA_thermal\models', f'mnist_mnistm_model1_epoch_{epoch}.pth'))
    source_accu.append(test_(source_dataset_name, epoch))
    target_accu.append(test_(target_dataset_name, epoch))

plt.plot(np.arange(0, n_epoch, 1), np.array(source_accu), label='Source Classification Accuracy')
plt.plot(np.arange(0, n_epoch, 1), np.array(target_accu), label='Target Classification Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
output_path = r'D:\UDA-Thermal\UDA_thermal\output\acc_alex_cbam_digit_final_.png'
plt.savefig(output_path)

print('done')
