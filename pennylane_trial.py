import pennylane as qml
import numpy as np
from pennylane.templates import RandomLayers
#import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tqdm import tqdm

import qiskit
from qiskit.circuit.library import QFT
from qiskit_textbook.widgets import scalable_circuit

n_epochs = 30
n_layers = 1
n_train = 50
n_test = 30
'''
SAVE_PATH = 'quanvolution/'
PREPROCESS = True
np.random.seed(0)
tf.random.set_seed(0)
'''
mnist_dataset = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist_dataset.load_data()

train_images = train_images[:n_train]
train_labels = train_labels[:n_train]
test_images = test_images[:n_test]
test_labels = test_labels[:n_test]

train_images = train_images/255
test_images = test_images/255


#--------- QFCNN stuff -----------

image_shape = train_images[0].shape #(28,28)

#print(train_images[0].flatten())

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
    thetas = flat_image*(np.pi/2)
    n_qubits = 11
    circuit = qiskit.QuantumCircuit(n_qubits)
    circuit.h(range(n_qubits-1))
    n = len(flat_image)
    circuit.x(n_qubits-2)
    circuit.mcry(thetas[0], q_controls = range(n_qubits-1), q_target = n_qubits-1)
    for i in tqdm(range(1,n)):
        circuit.x(n_qubits-2)
        for j in range(1,n_qubits-1):
            if i % 2**j:
                circuit.x(n_qubits-2-j)
        circuit.mcry(thetas[i], q_controls = range(n_qubits-1), q_target = n_qubits-1)
    return circuit

    
circuit = frqi(train_images[0])
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

#---------------------------------

'''
train_images = np.array(train_images[..., tf.newaxis])
test_images = np.array(test_images[..., tf.newaxis])

dev = qml.device('default.qubit', wires = 4)
rand_params = np.random.uniform(high = 2*np.pi, size = (n_layers, 4))

@qml.qnode(dev)
def circuit(phi):
    for j in range(4):
        qml.RY(np.pi*phi[j], wires = j)
    RandomLayers(rand_params, wires = list(range(4)))
    return [qml.expval(qml.PauliZ(j)) for j in range(4)]

def quanv(image):
    out = np.zeros((14,14,4))

    for j in range(0,28,2):
        for k in range(0,28,2):
            q_results = circuit(
                [
                    image[j,k,0],
                    image[j,k+1,0],
                    image[j+1,k,0],
                    image[j+1,k+1,0]
                ]
            )
            for c in range(4):
                out[j//2, k//2, c] = q_results[c]
    return out

if PREPROCESS == True:
    q_train_images = []
    for idx, img in tqdm(enumerate(train_images)):
        q_train_images.append(quanv(img))
    q_train_images = np.asarray(q_train_images)

    q_test_images = []
    for idx, img in tqdm(enumerate(test_images)):
        q_test_images.append(quanv(img))
    q_test_images = np.asarray(q_test_images)

def model():
    model = keras.models.Sequential(
        [
            keras.layers.Flatten(),
            keras.layers.Dense(10, activation = 'softmax')
        ]
    )
    model.compile(
        optimizer = 'adam',
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy']
    )
    return model

q_model = model()

q_history = q_model.fit(
    q_train_images,
    train_labels,
    validation_data = (q_test_images, test_labels),
    batch_size = 4,
    epochs = n_epochs,
    verbose = 2,
)
'''