import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchquantum as tq
import random
from torchquantum.datasets import MNIST
from torchquantum.encoding import encoder_op_list_name_dict
from torchquantum.layers import U3CU3Layer0
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import transforms
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as col
from qiskit import IBMQ
from qiskit.providers.ibmq import least_busy
from torchquantum.plugins import QiskitProcessor
from quanvolutional_filter import *

# Use IBMQ token
'''
IBMQ.save_account('c86d1b01559b0891d9d5a8b9d150ad6e1a0255a700736bdfcf05052b826b36f27970a449a5bded6425aded6bf8f5a13be1379258fb5f1d539015cb017dddfd35', overwrite=True)
IBMQ.load_account()
provider = IBMQ.get_provider(hub = 'ibm-q')
devices = provider.backends(simulator = False, operational = True)
backend = least_busy(devices)
processor = QiskitProcessor(use_real_qc = True, backend = backend, hub = 'ibm-q', group = 'open', project = 'main')
'''
IMG_SIZE = 16 # Must be less than or equal to 240, and must be divisible by kernel width
N_QUBITS = 4 # Must be a square number (square of kernel width)
N_EPOCHS = 30 

class HybridModel(torch.nn.Module):

    '''
    Combines quanvolutional filter with classical linear layers to make a hybrid model.
    '''

    def __init__(self):
        super().__init__()
        self.qf = QuanvolutionalFilter()
        self.linear = torch.nn.Linear(1*IMG_SIZE*IMG_SIZE, 2)

    def forward(self, x, use_qiskit = False):
        with torch.no_grad():
            x = self.qf(x, use_qiskit)
        x = x.reshape(-1, IMG_SIZE*IMG_SIZE)
        x = self.linear(x)
        return F.log_softmax(x, -1)

class HybridModel_without_qf(torch.nn.Module):

    '''
    Classical model that doesn't use quanvolutional filter.
    '''

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(IMG_SIZE*IMG_SIZE, 2)

    def forward(self, x, use_qiskit = False):
        x = x.view(-1, IMG_SIZE*IMG_SIZE)
        x = self.linear(x)
        return F.log_softmax(x, -1)

class Model1(torch.nn.Module):

    '''
    Trainable quanv layer + classical linear layer
    '''

    def __init__(self) -> None:
        super().__init__()
        self.qf = TrainableQuanvolutionalFilter()
        self.linear = torch.nn.Linear(1*self.qf.out_dim*self.qf.out_dim,2)

    def forward(self, x, use_qiskit = False):
        x = x.view(-1, IMG_SIZE, IMG_SIZE)
        x = self.qf(x)
        x = x.reshape(-1, 1*self.qf.out_dim*self.qf.out_dim)
        x = self.linear(x)
        return F.log_softmax(x, -1)

class Classical1(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 5, kernel_size = 3)
        self.linear = torch.nn.Linear(605,2)

    def forward(self, x, use_qiskit = False):
        x = x.view(-1, IMG_SIZE, IMG_SIZE)
        bsz = 1
        x = F.avg_pool2d(x, 3).view(bsz, 24, 24)
        x = x.view(bsz, 24, 24)
        x = F.relu(self.conv(x))
        x = F.max_pool2d(x,2)
        x = x.reshape(-1, 605)
        x = self.linear(x)
        return F.log_softmax(x, -1)


class Model2(torch.nn.Module):

    '''
    Trainable quanv layer + quantum fully connected layer
    '''

    def __init__(self) -> None:
        super().__init__()
        self.qf = TrainableQuanvolutionalFilter(IMG_SIZE)
        self.qfc = QFC()

    def forward(self, x, use_qiskit = False):
        x = x.view(-1, IMG_SIZE, IMG_SIZE)
        x = self.qf(x)
        x = x.reshape(-1, 16)
        x = self.qfc(x)
        return F.log_softmax(x, -1)

class Model3(torch.nn.Module):

    '''
    Quantum classifier
    '''

    def __init__(self) -> None:
        super().__init__()
        self.qfc = QuantumClassifier()

    def forward(self, x, use_qiskit = False):
        x = self.qfc(x)
        return F.log_softmax(x, -1)

class Classical3(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.linear1 = torch.nn.Linear(16*16, 32)
        self.linear2 = torch.nn.Linear(32, 2)

    def forward(self, x, use_qiskit = False):
        bsz = 1
        #x = F.avg_pool2d(x, 3).view(bsz, 16, 16)
        #x = F.avg_pool2d(x, 4).view(bsz, 16)
        x = F.avg_pool2d(x, 3).view(bsz, 16*16)
        x = self.linear1(x)
        x = self.linear2(x)
        return F.log_softmax(x, -1)

class FinalModel(torch.nn.Module):

    def __init__(self, skip) -> None:
        super().__init__()
        self.skip = skip
        self.qf1 = TrainableQuanvolutionalFilter(N_QUBITS, IMG_SIZE, pool = skip)
        self.k = np.sqrt(N_QUBITS).astype(int)
        if skip:
            self.s = np.floor((IMG_SIZE)/self.k).astype(int)
        else:
            self.s = np.floor((IMG_SIZE-1)/self.k).astype(int)
        self.linear1 = torch.nn.Linear(N_QUBITS*self.s*self.s, 2)

    def forward(self, x, use_qiskit = False):
        x = x.view(-1, IMG_SIZE, IMG_SIZE)
        x = self.qf1(x, use_qiskit = use_qiskit)
        if not self.skip:
            x = F.avg_pool2d(x, self.k)#.view(N_QUBITS, self.s, self.s)
        x = x.reshape(-1, N_QUBITS*self.s*self.s)
        x = self.linear1(x)
        return F.log_softmax(x, -1)

class ClassicalTrial(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, N_QUBITS, kernel_size = int(np.sqrt(N_QUBITS)))
        s = int(np.floor((IMG_SIZE-1)/np.sqrt(N_QUBITS)))
        self.out_dim = N_QUBITS*s*s
        self.linear1 = torch.nn.Linear(self.out_dim, 2)

    def forward(self, x, use_qiskit = False):
        x = x.view(-1, IMG_SIZE, IMG_SIZE)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, int(np.sqrt(N_QUBITS)))
        x = x.reshape(-1, self.out_dim)
        x = self.linear1(x)
        return F.log_softmax(x, -1)

# Training subroutine
def train(dataloader, model, device, optimizer):
    target_all = []
    output_all = []
    for data, label in dataloader:
        inputs = data.to(device)
        targets = label.to(device)

        outputs = model(inputs)
        loss = F.nll_loss(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'loss: {loss.item()}', end = '\r')
        target_all.append(targets)
        output_all.append(outputs)
    target_all = torch.cat(target_all, dim = 0)
    output_all = torch.cat(output_all, dim = 0)
    _, indices = output_all.topk(1, dim = 1)
    masks = indices.eq(target_all.view(-1,1).expand_as(indices))
    size = target_all.shape[0]
    corrects = masks.sum().item()
    accuracy = corrects / size
    print('Training set accuracy: {}'.format(accuracy))

# Validation testing function
def valid_test(dataloader, split, model, device, qiskit = False):
    target_all = []
    output_all = []
    with torch.no_grad():
        for data, label in dataloader:
            inputs = data.to(device)
            targets = label.to(device)
            
            outputs = model(inputs, use_qiskit = qiskit)

            target_all.append(targets)
            output_all.append(outputs)
        target_all = torch.cat(target_all, dim = 0)
        output_all = torch.cat(output_all, dim = 0)

    _, indices = output_all.topk(1, dim = 1)
    masks = indices.eq(target_all.view(-1,1).expand_as(indices))
    size = target_all.shape[0]
    corrects = masks.sum().item()
    accuracy = corrects / size
    loss = F.nll_loss(output_all, target_all).item()
    print(f'{split} set accuracy: {accuracy}')
    print(f'{split} set loss: {loss}')
    return accuracy, loss

def run(quanv_model, device, test_loader, train_loader, progress = False):
    model = quanv_model.to(device) #HybridModel().to(device)
    #model_without_qf = HybridModel_without_qf().to(device)
    n_epochs = N_EPOCHS
    optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

    accu_list1 = []
    loss_list1 = []
    filter_set = []

    for epoch in tqdm(range(1, n_epochs+1)):
        # Train
        print(f'Epoch {epoch}:')
        train(train_loader, model, device, optimizer)
        print(optimizer.param_groups[0]['lr'])
        if progress:
            if epoch == 1 or epoch % 10 == 0:
                filters = model.qf1(x_train[12][0])
                filter_set.append(filters)
        # Validation test

        accu, loss = valid_test(test_loader, 'test', model, device, qiskit = False)
        accu_list1.append(accu)
        loss_list1.append(loss)
        scheduler.step()
    #torch.save(model, 'model3_v1.pt')
    if progress:
        return accu_list1, loss_list1, filter_set
    return accu_list1, loss_list1

if __name__ == '__main__':

    x_train = CustomImageDataset(
        annotations_file = 'brain_cancer_output/val.csv',
        img_dir = 'brain_cancer_output/val/Brain Tumor/',
        transform = transforms.Compose([transforms.ToPILImage(), transforms.Grayscale(), transforms.Resize(size = IMG_SIZE), transforms.ToTensor()])
    )

    x_test = CustomImageDataset(
        annotations_file = 'brain_cancer_output/test.csv',
        img_dir = 'brain_cancer_output/test/Brain Tumor/',
        transform = transforms.Compose([transforms.ToPILImage(), transforms.Grayscale(), transforms.Resize(size = IMG_SIZE), transforms.ToTensor()])
    )

    train_loader = torch.utils.data.DataLoader(x_train, batch_size = 1, shuffle = True) # 376 items
    test_loader = torch.utils.data.DataLoader(x_test, batch_size = 1, shuffle = True) # 753 items

    # Define device, models, optimizer, scheduler and number of epochs
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    c_acc, c_loss = run(ClassicalTrial(), device, test_loader, train_loader)
    q1_acc, q1_loss, filters = run(FinalModel(skip = True), device, test_loader, train_loader, progress = True)
    #q2_acc, q2_loss = run(FinalModel(skip = False), device, test_loader, train_loader)
   
    # Plot filters over time
    f, axarr = plt.subplots(filters[0].size(0), len(filters))
    epoch_list = [1,10,20,30]
    for k, fil in enumerate(filters):
        norm = col.Normalize(vmin=0,vmax=1)
        for c in range(fil.size(0)):
            axarr[c,0].set_ylabel('Channel {}'.format(c))
            axarr[0,k].set_title('Epoch {}'.format(epoch_list[k]))
            if k != 0:
                axarr[c,k].yaxis.set_visible(False)
            img = fil[c,:].detach().numpy()
            axarr[c,k].imshow(img, norm=norm, cmap='gray')
    plt.savefig('imgs3b.png')
    plt.show()
    plt.close()
    '''
    norm = col.Normalize(vmin=0,vmax=1)
    og_img = x_train[12][0].squeeze().detach().numpy()
    plt.imshow(og_img, norm = norm, cmap = 'gray')
    plt.savefig('og_img33.png')
    plt.show()
    plt.close()
    
    # Accuracy plot
    plt.plot(c_acc, label = 'Classical')
    plt.plot(q1_acc, label = 'Quanv with stride')
    #plt.plot(q2_acc, label = 'Quanv without stride')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('acc3b_plot.png')
    plt.show()
    plt.close()

    # Loss plot
    plt.plot(c_loss, label = 'Classical')
    plt.plot(q1_loss, label = 'Quantum')
    #plt.plot(q2_loss, label = 'Quanv without stride')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss3b_plot.png')
    plt.show()
    '''
    #print('Classical: ', c_acc[-1])
    print('Quanv with stride: ', q1_acc[-1])
    #print('Quanv without stride: ', q2_acc[-1])'''