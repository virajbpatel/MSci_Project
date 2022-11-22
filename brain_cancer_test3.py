import pandas as pd
import os
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torch.autograd import Function
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import qiskit
from qiskit import transpile, assemble, BasicAer
from qiskit.visualization import *
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.neural_networks import TwoLayerQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.opflow import AerPauliExpectation, PauliSumOp
from qiskit.utils import QuantumInstance

#df = pd.read_csv('data/brain_cancer/Brain Tumor.csv')
'''
df['Image'] = [s + '.jpg' for s in df['Image'].values.tolist()]

df = df[['Image', 'Class']]
df.to_csv('data/brain_cancer/labels.csv')
'''
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

class QuantumCircuit:

    def __init__(self, n_qubits, backend, shots) -> None:
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        all_qubits = [i for i in range(n_qubits)]
        self.theta = qiskit.circuit.Parameter('theta')

        self._circuit.rx(np.pi/2, all_qubits)
        self._circuit.ry(self.theta, all_qubits)
        self._circuit.cx(0,1)
        self._circuit.cx(1,2)
        self._circuit.cx(2,3)
        self._circuit.ry(self.theta, all_qubits)
        self._circuit.cx(0,1)
        self._circuit.cx(1,2)
        self._circuit.cx(2,3)
        self._circuit.measure_all()

        self.backend = backend
        self.shots = shots

    def run(self, thetas):
        t_qc = transpile(self._circuit, self.backend)
        qobj = assemble(t_qc, shots = self.shots, parameter_binds = [{self.theta: theta} for theta in thetas])
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)

        probabilities = counts / self.shots
        expectation = np.sum(states * probabilities)

        return np.array([expectation])

class HybridFunction(Function):

    @staticmethod
    def forward(ctx, input, quantum_circuit, shift):
        ctx.shift = shift
        ctx.quantum_circuit = quantum_circuit
        expectation_z = ctx.quantum_circuit.run(input[0].tolist())
        result = torch.tensor(np.array([expectation_z]))
        ctx.save_for_backward(input, result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        input, expectation_z = ctx.saved_tensors
        input_list = np.array(input.tolist())

        shift_right = input_list + np.ones(input_list.shape) * ctx.shift
        shift_left = input_list - np.ones(input_list.shape) * ctx.shift

        gradients = []
        for i in range(len(input_list)):
            expectation_right = ctx.quantum_circuit.run(shift_right[i])
            expectation_left = ctx.quantum_circuit.run(shift_left[i])
            gradient = torch.tensor(np.array([expectation_right])) - torch.tensor(np.array([expectation_left]))
            gradients.append(gradient)
        gradients = np.array(gradients).T
        return torch.tensor(np.array([gradients])).float() * grad_output.float(), None, None

class Hybrid(nn.Module):
    
    def __init__(self, backend, shots, shift) -> None:
        super(Hybrid, self).__init__()
        self.quantum_circuit = QuantumCircuit(4, backend, shots)
        self.shift = shift

    def forward(self, input):
        return HybridFunction.apply(input, self.quantum_circuit, self.shift)

qi = QuantumInstance(BasicAer.get_backend('qasm_simulator'), shots = 1024)

basis_gates = ['barrier', 'sx', 'id', 'u3', 'measure', 'reset', 'u1', 'snapshot', 'x', 'unitary', 'rz', 'delay', 'u2', 'cx']
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel
from qiskit.transpiler.passes import BasisTranslator
bt_pass = BasisTranslator(sel, basis_gates)


def create_qnn():
    feature_map = ZZFeatureMap(2)
    ansatz = RealAmplitudes(2, reps = 1)
    qnn = TwoLayerQNN(
        2,
        feature_map,
        ansatz,
        input_gradients=True,
        observable = PauliSumOp.from_list([('Z'*2, 1)]),
        exp_val = AerPauliExpectation(),
        quantum_instance = qi
    )
    return qnn

class Net(nn.Module):

    def __init__(self, qnn) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 12, kernel_size = 5)
        self.conv2 = nn.Conv2d(12, 16, kernel_size = 5)
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(51984, 64)
        self.fc2 = nn.Linear(64, 2)
        self.qnn = TorchConnector(qnn)
        self.fc3 = nn.Linear(1,1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = x.view(x.shape[0], -1)#x.view(1, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.qnn(x)
        x = self.fc3(x)
        return torch.cat((x, 1-x), -1)


x_train = CustomImageDataset(
    annotations_file = 'brain_cancer_output/val.csv',
    img_dir = 'brain_cancer_output/val/Brain Tumor/',
    transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
)

x_test = CustomImageDataset(
    annotations_file = 'brain_cancer_output/test.csv',
    img_dir = 'brain_cancer_output/test/Brain Tumor/',
    transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
)

n_samples = 100
#idx = np.append(np.where(x_train.targets == 0)[0][:n_samples], np.where(x_train.targets == 1)[0][:n_samples])
#x_train.data = x_train.data[idx]
#x_train.targets = x_train.targets[idx]
train_loader = torch.utils.data.DataLoader(x_train, batch_size = 1, shuffle = True)
test_loader = torch.utils.data.DataLoader(x_test, batch_size = 1, shuffle = True)

qnn = create_qnn()
model = Net(qnn)
optimizer = optim.Adam(model.parameters(), lr = 0.001)
loss_func = nn.NLLLoss()
print('Model created')

epochs = 20
loss_list = []

print('Training Model')
model.train()
print('model.train() completed')
for epoch in range(epochs):
    total_loss = []
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())
    loss_list.append(sum(total_loss)/len(total_loss))
    print('Training [{:.0f}%]\tLoss: {:.4f}'.format(100.*(epoch + 1)/epochs, loss_list[-1]))

model.eval()
with torch.no_grad():
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        output = model(data)
        pred = output.argmax(dim = 1, keepdim = True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        loss = loss_func(output, target)
        total_loss.append(loss.item())

    print('Performance on test data:\n\tLoss: {:.4f}\n\tAccuracy: {:.1f}%'.format(
        sum(total_loss)/len(total_loss),
        correct/len(test_loader)*100)
        )