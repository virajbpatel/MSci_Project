#--------- QFCNN stuff -----------

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import transforms
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import qiskit
from qiskit.circuit.library import QFT
from qiskit_textbook.widgets import scalable_circuit

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

x_train = CustomImageDataset(
    annotations_file = 'brain_cancer_output/val.csv',
    img_dir = 'brain_cancer_output/val/Brain Tumor/',
    transform = transforms.Compose([transforms.ToPILImage(), transforms.Grayscale(), transforms.ToTensor()])
)

x_test = CustomImageDataset(
    annotations_file = 'brain_cancer_output/test.csv',
    img_dir = 'brain_cancer_output/test/Brain Tumor/',
    transform = transforms.Compose([transforms.ToPILImage(), transforms.Grayscale(), transforms.ToTensor()])
)

train_loader = torch.utils.data.DataLoader(x_train, batch_size = 1, shuffle = True)
test_loader = torch.utils.data.DataLoader(x_test, batch_size = 1, shuffle = True)

sample_image = torch.squeeze(x_train[0][0]).cpu().detach().numpy()

image_shape = sample_image[0].shape #(28,28)

# QFT code: https://qiskit.org/textbook/ch-algorithms/quantum-fourier-transform.html

# Quantum circuit for rotation blocks in QFT
def qft_rotations(circuit, n):
    # No qubits => no circuit
    if n == 0:
        return circuit
    # Reduce by 1 because of Python indexing
    n -= 1
    # Superpose last qubit
    circuit.h(n)
    for qubit in range(n):
        # Rotate by pi/(2^i) for i=1 to n
        circuit.cp(np.pi/2**(n-qubit), qubit, n)
    # Recursive function
    qft_rotations(circuit, n)

scalable_circuit(qft_rotations)

# Quantum circuit for swap operations in QFT
def swap_registers(circuit, n):
    # Swaps qubits outside to inside
    for qubit in range(n//2):
        circuit.swap(qubit, n-qubit-1)
    return circuit

# Function that performs QFT given an initial circuit and n qubits
def qft(circuit, n):
    qft_rotations(circuit, n)
    swap_registers(circuit, n)
    return circuit

scalable_circuit(qft)

# Function that performs inverse QFT given an initial circuit and n qubits
def inverse_qft(circuit, n):
    qft_circuit = qft(qiskit.QuantumCircuit(n), n)
    invqft_circuit = qft_circuit.inverse()
    circuit.append(invqft_circuit, circuit.qubits[:n])
    return circuit.decompose()

def prep_image(image):
    # Flatten image matrix (M,L) into a vector (length = ML)
    flat_image = image.flatten()
    # Find norm of flattened image
    mag = np.linalg.norm(flat_image)
    # Find values of c_k in FRQI representation
    thetas = flat_image/mag
    n_qubits = 11
    circuit = qiskit.QuantumCircuit(n_qubits)
    for qubit in range(n_qubits-1):
        circuit.h(qubit)

def frqi(image):
    flat_image = image.flatten()
    mag = np.linalg.norm(flat_image)
    thetas = flat_image*(np.pi/2)/mag
    n_qubits = 3
    circuit = qiskit.QuantumCircuit(n_qubits)
    circuit.h(range(n_qubits-1))
    circuit.barrier()
    circuit.x(n_qubits-2)
    n = len(thetas)
    circuit.mcry(thetas[0], q_controls = list(range(n_qubits-1)), q_target = n_qubits-1)
    circuit.barrier()
    for i in range(1,n):
        circuit.x(n_qubits-2)
        for j in range(1, n_qubits-1):
            if i % 2**j:
                circuit.x(n_qubits-2-j)
        circuit.mcry(thetas[i], q_controls = list(range(n_qubits-1)), q_target = n_qubits-1)
        circuit.barrier()
    return circuit


img = np.array([[1,2],[2,1]])
circuit = frqi(img)
circuit.draw('mpl')
plt.show()

'''    
circuit = frqi(sample_image)
circuit = qft(circuit, 17)
print('Circuit completed')
circuit.measure_all()
print('Measurement taken')
backend = qiskit.BasicAer.get_backend('qasm_simulator')
t = qiskit.transpile(circuit, backend)
print('Transpiled')
qobj = qiskit.assemble(t, shots = 4096)
print('Assembled')
result = backend.run(qobj).result()
print('Run completed')
counts = result.get_counts(circuit)
qiskit.visualization.plot_histogram(counts)
plt.show()
'''
#---------------------------------