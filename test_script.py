import os
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision import transforms
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torchvision.transforms as T

device_gpu = 0

def test_(dataset_name, epoch):
    assert dataset_name in ['source', 'target']

    model_root = os.path.join(r"D:\UDA-Thermal\UDA_thermal\models", 'models')
    
    cuda = True
    cudnn.benchmark = True
    batch_size = 32
    image_size = 224
    alpha = 0

    img_rgb_resized = np.load(r"D:\UDA-Thermal\UDA_thermal\archive\X_rgb_224.npy")
    labels_rgb = np.load(r"D:\UDA-Thermal\UDA_thermal\archive\Y_rgb_224.npy")
    img_th_rot = np.load(r"D:\UDA-Thermal\UDA_thermal\archive\X_th_224-001.npy")
    labels_th = np.load(r"D:\UDA-Thermal\UDA_thermal\archive\Y_th_224.npy")

    img_rgb_train, img_rgb_test, labels_rgb_train, labels_rgb_test = train_test_split(
        img_rgb_resized, labels_rgb, test_size=0.1, random_state=42, stratify=labels_rgb)

    img_th_train, img_th_test, labels_th_train, labels_th_test = train_test_split(
        img_th_rot, labels_th, test_size=0.6, random_state=42, stratify=labels_th)

    train = torch.utils.data.TensorDataset(
        torch.from_numpy(img_rgb_train), torch.from_numpy(labels_rgb_train))
    test = torch.utils.data.TensorDataset(
        torch.from_numpy(img_rgb_test), torch.from_numpy(labels_rgb_test))
    train_dataloader_source = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    test_dataloader_source = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=True)

    train_th = torch.utils.data.TensorDataset(
        torch.from_numpy(img_th_train), torch.from_numpy(labels_th_train))
    test_th = torch.utils.data.TensorDataset(
        torch.from_numpy(img_th_test), torch.from_numpy(labels_th_test))
    train_dataloader_target = torch.utils.data.DataLoader(train_th, batch_size=batch_size, shuffle=True)
    test_dataloader_target = torch.utils.data.DataLoader(test_th, batch_size=batch_size, shuffle=True)

    if dataset_name == 'source':
        dataloader = test_dataloader_source
    elif dataset_name == 'target':
        dataloader = test_dataloader_target

    """ training """

    my_net = torch.load(os.path.join(r'D:\UDA-Thermal\UDA_thermal\models', f'mnist_mnistm_model1_epoch_{epoch}.pth'))
    my_net = my_net.eval()

    if cuda:
        my_net = my_net.cuda(device_gpu)

    len_dataloader = len(dataloader)
    data_target_iter = iter(dataloader)

    i = 0
    n_total = 0
    n_correct = 0

    while i < len_dataloader:

        # test model using target data
        data_target = next(data_target_iter)
        t_img, t_label = data_target

        batch_size = len(t_label)

        input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
        class_label = torch.LongTensor(batch_size)

        if cuda:
            t_img = t_img.cuda(device_gpu)
            t_label = t_label.cuda(device_gpu)
            input_img = input_img.cuda(device_gpu)
            class_label = class_label.cuda(device_gpu)

        input_img.resize_as_(t_img).copy_(t_img)
        class_label.resize_as_(t_label).copy_(t_label)

        class_output, _ = my_net(input_data=input_img, alpha=alpha)
        pred = class_output.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(class_label.data.view_as(pred)).cpu().sum()
        n_total += batch_size

        i += 1

    accu = n_correct.data.numpy() * 1.0 / n_total
    print(f'epoch: {epoch}, accuracy of the {dataset_name} dataset: {accu:.6f}')
    return accu
