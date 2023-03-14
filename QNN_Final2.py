import torch
from torch.autograd import Function
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchmetrics

import qiskit
from qiskit import execute, transpile, assemble, BasicAer, QuantumRegister, ClassicalRegister, Aer
from qiskit.circuit import Parameter,QuantumCircuit
from qiskit.visualization import *
# from qiskit_machine_learning.connectors import TorchConnector
from qiskit.opflow import AerPauliExpectation, PauliSumOp
from qiskit.utils import QuantumInstance

import quanvolutional_filter as quanv

import pandas as pd
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

#This part is only needed for my Mac
#import os 
#os.environ['KMP_DUPLICATE_LIB_OK']='True'

"""
For loading images into a proper dataset 
"""
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform = None, target_transform = None) -> None:
        self.img_labels = pd.read_csv(annotations_file, index_col = [0])
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx,0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx,1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

"""
For some tensor manipulation
"""
def to_numbers(tensor_list):
    num_list = []
    for tensor in tensor_list:
        num_list += [tensor.item()]
    return num_list

class QFC_Net(torch.nn.Module):

    def __init__(self, N_QUBITS,N_CIRCUITS) -> None:
        self.nqubits = N_QUBITS
        self.ncircuits = N_CIRCUITS
        self.inputdim = 16
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(43264,self.inputdim*self.ncircuits)
        self.qfc = quanv.QFC(self.nqubits, input_dim = self.inputdim, encoding = '4x4_ryzxy')
        self.out = nn.Linear(self.ncircuits*self.nqubits, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv_drop(self.conv3(x)), 2))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        y=torch.empty((1,4), dtype=torch.int64)
        for i in range(0,len(x[0]),self.inputdim):
            qs=x[0][i:i+self.inputdim]
            qs = qs.reshape(-1, self.inputdim)
            z=self.qfc(qs) #outputs 4 numbers
            result,ind=z.topk(4)
            y = torch.cat((y,result),dim=0)
        x=y[1:]
        x = x.reshape(-1, self.ncircuits*self.nqubits)  
        x=self.out(x)
        return F.log_softmax(x,dim=1)

"""
Putting everything together in 1 class to perform training, testing, and saving of data/models
train_loader,val_loader,test_loader not defined in this file, not sure if we should rename these here
"""
class quantum_model():
    def __init__(self, N_QUBITS, N_CIRCUITS, LR, MOM, train_data, val_data, test_data):
        self.NQUBITS = N_QUBITS
        self.NCIRCUITS = N_CIRCUITS
        self.LR = LR
        self.MOM = MOM
        self.trainset = train_data
        self.valset = val_data
        self.testset = test_data

        self.model = QFC_Net(self.NQUBITS, self.NCIRCUITS) 
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.LR,
                        momentum=self.MOM,nesterov=True)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=0.01, weight_decay=1e-4)
                    
        self.loss = nn.NLLLoss()

        self.loss_list_quantum = []
        self.pq = []
        self.tq = []

    def train(self,EPOCH,iteration):
        self.EPOCH = EPOCH
        self.model.train()
        self.accuracy_val=[]
        for epoch in range(self.EPOCH):
            target_all = []
            output_all = []
            count=0
            total_loss = []
            correct=0
            for batch_idx, (data, target) in enumerate(tqdm(self.trainset, position=0, leave=True)):
                self.optimizer.zero_grad()        
                output = self.model(data)
                loss = self.loss(output, target)
                loss.backward()
                self.optimizer.step()
                total_loss.append(loss.item())
            
            self.corr_val=0
            self.loss_val=[]
            for batch_idx, (data, target) in enumerate(self.valset):
                val_out = self.model(data)  
                pred = val_out.argmax(dim = 1, keepdim = True)
                self.corr_val += pred.eq(target.view_as(pred)).sum().item()
                loss = self.loss(val_out, target)
                self.loss_val.append(loss.item())
            self.accuracy_quantum_val=self.corr_val/len(self.valset)*100
            self.accuracy_val.append(self.accuracy_quantum_val)
            self.loss_list_quantum.append(sum(self.loss_val)/len(self.loss_val))
            print('Quantum Training [{:.0f}%]\t Val Loss: {:.4f} \t Val Accuracy: {:.1f}'.format(100. * (epoch + 1) / self.EPOCH, self.loss_list_quantum[-1], self.accuracy_quantum_val))
        plt.ioff()
        plt.title("Quantum Classifier{}a".format(iteration))
        plt.plot(self.loss_list_quantum,'.')
        plt.grid()
        plt.savefig("Quantum Classifier{}a.jpg".format(iteration))
        plt.close()

    def eval(self):
        self.model.eval()
        with torch.no_grad():
            correct = 0
            total_loss = []
            for batch_idx, (data, target) in enumerate(self.testset):
                output = self.model(data)
                pred = output.argmax(dim = 1, keepdim = True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                self.correct=correct
                loss = self.loss(output, target)
                total_loss.append(loss.item())
            self.loss_quantum=sum(total_loss)/len(total_loss)
            self.accuracy_quantum=correct/len(self.testset)*100
            print('Quantum Classifier performance on test data:\n\tLoss: {:.4f}\n\tAccuracy: {:.1f}%'.format(
                self.loss_quantum,
                self.accuracy_quantum)
                )

    def save(self,iteration):
        torch.save(self.model,"Models/QC_A_{:.1f}__{}a.pt".format(self.correct/len(self.testset)*100,iteration))
        np.savetxt("data/AccuracyListQC{}a.csv".format(iteration),self.accuracy_val)
        np.savetxt("data/LossListQC{}a.csv".format(iteration),self.loss_list_quantum)



