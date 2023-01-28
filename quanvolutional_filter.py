# To be run in Python 3.7, all other files work with Python 3.9

# Import releveant packages
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

IMG_SIZE = 100 # Must be less than or equal to 240, and must be divisible by kernel width
N_QUBITS = 4 # Must be a square number (square of kernel width)
N_EPOCHS = 25 

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

class QuanvolutionalFilter(tq.QuantumModule):

    '''
    Variational quantum circuit to pass over an image, analogous to classical convolutional filter. Performs 
    complex operations in a high dimensional Hilbert space, and so has more potential than the classical 
    Frobenius inner product.
    '''

    # Initialise a 4-qubit quantum circuit to encode pixels in, and then perform quanvolution with layers of 
    # random gates
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = N_QUBITS
        self.q_device = tq.QuantumDevice(n_wires = self.n_wires)
        encoding_list = []
        for i in range(self.n_wires):
            encoding = {'input_idx': [i], 'func': 'ry', 'wires': [i]}
            encoding_list.append(encoding)
        self.encoder = tq.GeneralEncoder(encoding_list)
        self.q_layer = tq.RandomLayer(n_ops = 8, wires = list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)
    
    # Pass VQC over image
    def forward(self, x, use_qiskit = False):
        bsz = 1
        size = IMG_SIZE
        x = x.view(bsz, size, size)
        data_list = []
        # Filter is 2x2 pixels, transpose tensor so that it transforms as follows:
        '''
        [                          [
            [r1,g1,b1],                 [r1,r2,r3,r4],
            [r2,g2,b2],     ->          [g1,g2,g3,g4],
            [r3,g3,b3],                 [b1,b2,b3,b4]
            [r4,g4,b4]             ]
        ]                          
        '''
        step = int(np.sqrt(self.n_wires))
        for c in range(0, size, step):
            for r in range(0, size, step):
                pixels = []
                for i in range(step):
                    for j in range(step):
                        pixels.append(x[:,c+i,r+j])
                data = torch.transpose(torch.cat(tuple(pixels)).view(self.n_wires,bsz), 0, 1) #(x[:,c,r], x[:,c,r+1], x[:,c+1,r], x[:,c+1,r+1])
                if use_qiskit:
                    data = self.qiskit_processor.process_parameterized(
                        self.q_device, self.encoder, self.q_layer, self.measure, data)
                else:
                    self.encoder(self.q_device, data)
                    self.q_layer(self.q_device)
                    data = self.measure(self.q_device)

                data_list.append(data.view(bsz, self.n_wires))

        result = torch.cat(data_list, dim = 1).float()

        return result

class TrainableQuanvolutionalFilter(tq.QuantumModule):

    def __init__(self, in_dim) -> None:
        super().__init__()
        self.n_wires = N_QUBITS
        self.q_device = tq.QuantumDevice(n_wires = self.n_wires)
        encoding_list = []
        for i in range(self.n_wires):
            encoding = {'input_idx': [i], 'func': 'ry', 'wires': [i]}
            encoding_list.append(encoding)
        self.encoder = tq.GeneralEncoder(encoding_list)
        self.arch = {'n_wires': self.n_wires, 'n_blocks': 5, 'n_layers_per_block': 2}
        self.q_layer = U3CU3Layer0(self.arch)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.kernel_size = int(np.sqrt(N_QUBITS))
        self.out_dim = np.floor((in_dim - self.kernel_size)/self.kernel_size + 1).astype(int)

    def forward(self, x, use_qiskit = False):
        bsz = 1
        #x = F.avg_pool2d(x, self.kernel_size).view(bsz, self.out_dim, self.out_dim)
        #x = x.view(bsz, self.out_dim, self.out_dim)

        size = IMG_SIZE #self.out_dim
        stride = int(np.sqrt(self.n_wires))
        x = x.view(bsz, size, size)

        data_list = []

        for c in range(0, size-1, stride):
            row = []
            for r in range(0, size-1, stride):
                pixels = []
                for i in range(stride):
                    for j in range(stride):
                        pixels.append(x[:,c+i,r+j])
                data = torch.transpose(torch.cat(tuple(pixels)).view(self.n_wires,bsz), 0, 1)
                if use_qiskit:
                    data = self.qiskit_processor.process_parameterised(
                        self.q_device, self.encoder, self.q_layer, self.measure, data
                    )
                else:
                    self.encoder(self.q_device, data)
                    self.q_layer(self.q_device)
                    data = self.measure(self.q_device)

                row.append(data.view(bsz, self.n_wires))
            data_list.append(torch.stack(row))
        data_list = torch.stack(data_list)
        data_list = torch.transpose(torch.squeeze(data_list), 0, 2).float()

        result = data_list
        return result

class QuantumClassifier(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = N_QUBITS
        self.q_device = tq.QuantumDevice(n_wires = self.n_wires)
        self.encoder = tq.GeneralEncoder(encoder_op_list_name_dict['4x4_ryzxy'])
        self.arch = {'n_wires': self.n_wires, 'n_blocks': 8, 'n_layers_per_block': 2}
        self.ansatz = U3CU3Layer0(self.arch)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.kernel_size = 3
        self.out_dim = np.floor((IMG_SIZE - self.kernel_size)/self.kernel_size + 1).astype(int)

    def forward(self, x, use_qiskit = False):
        bsz = 1
        #x = F.avg_pool2d(x, 6).view(bsz, 16)

        x = F.avg_pool2d(x, self.kernel_size).view(bsz, self.out_dim, self.out_dim)
        x = x.view(bsz, self.out_dim, self.out_dim)

        if use_qiskit:
            x = self.qiskit_processor.process_parameterised(
                self.q_device, self.encoder, self.q_layer, self.measure, x
            )
        else:
            self.encoder(self.q_device, x)
            self.q_layer(self.q_device)
            x = self.measure(self.q_device)
        return x

class QFC(tq.QuantumModule):

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = N_QUBITS
        self.q_device = tq.QuantumDevice(n_wires = self.n_wires)
        self.encoder = tq.GeneralEncoder(encoder_op_list_name_dict['4x4_ryzxy'])
        self.arch = {'n_wires': self.n_wires, 'n_blocks': 4, 'n_layers_per_block': 2}
        self.q_layer = U3CU3Layer0(self.arch)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x, use_qiskit = False):
        bsz = 1
        data = x
        if use_qiskit:
            data = self.qiskit_processor.process_parameterised(
                self.q_device, self.encoder, self.q_layer, self.measure, data
            )
        else:
            self.encoder(self.q_device, data)
            self.q_layer(self.q_device)
            data = self.measure(self.q_device)
        return data

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

    def __init__(self) -> None:
        super().__init__()
        self.qf1 = TrainableQuanvolutionalFilter(IMG_SIZE)
        #self.conv = torch.nn.Conv2d(1, , kernel_size = 3)
        #self.qf2 = TrainableQuanvolutionalFilter(self.qf1.out_dim)
        self.k = 2
        self.s = np.floor((IMG_SIZE)/self.k**2).astype(int)
        self.dropout = torch.nn.Dropout()
        self.linear1 = torch.nn.Linear(N_QUBITS*self.s*self.s,16)
        self.linear2 = torch.nn.Linear(16,2)

    def forward(self, x, use_qiskit = False):
        x = x.view(-1, IMG_SIZE, IMG_SIZE)
        x = self.qf1(x)
        x = F.avg_pool2d(x, self.k)#.view(N_QUBITS, self.s, self.s)
        x = self.dropout(x)
        #x = self.qf2(x)
        x = x.reshape(-1, N_QUBITS*self.s*self.s)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return F.log_softmax(x, -1)

class ClassicalTrial(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size = 4)
        self.dropout = torch.nn.Dropout()
        self.linear1 = torch.nn.Linear(10*16*16, 2)

    def forward(self, x, use_qiskit = False):
        x = x.view(-1, IMG_SIZE, IMG_SIZE)
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 6)
        x = self.dropout(x)
        x = x.reshape(-1, 10*16*16)
        x = self.linear1(x)
        return F.log_softmax(x, -1)


# Training subroutine
def train(dataloader, model, device, optimizer):
    for data, label in dataloader:
        inputs = data.to(device)
        targets = label.to(device)

        outputs = model(inputs)
        loss = F.nll_loss(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'loss: {loss.item()}', end = '\r')

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
    model = ClassicalTrial().to(device) #HybridModel().to(device)
    #model_without_qf = HybridModel_without_qf().to(device)
    n_epochs = N_EPOCHS
    optimizer = optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

    accu_list1 = []
    loss_list1 = []
    accu_list2 = []
    loss_list2 = []

    for epoch in tqdm(range(1, n_epochs+1)):
        # Train
        print(f'Epoch {epoch}:')
        train(train_loader, model, device, optimizer)
        print(optimizer.param_groups[0]['lr'])
        # Validation test
        accu, loss = valid_test(test_loader, 'test', model, device,)
        accu_list1.append(accu)
        loss_list1.append(loss)
        scheduler.step()
    #torch.save(model, 'model3_v1.pt')