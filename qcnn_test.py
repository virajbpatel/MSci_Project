from cmath import exp
from dbm.gnu import open_flags
from qiskit import BasicAer
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.circuit.library import ZFeatureMap
from qiskit.opflow import PauliSumOp, AerPauliExpectation
from qiskit_machine_learning.neural_networks import TwoLayerQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit.algorithms.optimizers import COBYLA
import numpy as np
#import matplotlib.pyplot as plt

'''
from qiskit.algorithms.optimizers import COBYLA
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.opflow import AerPauliExpectation, PauliSumOp
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_machine_learning.neural_networks import TwoLayerQNN
'''

def conv_circuit(params):
    target = QuantumCircuit(2)
    target.rz(-np.pi/2, 1)
    target.cx(1,0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0,1)
    target.ry(params[2], 1)
    target.cx(1,0)
    target.rz(np.pi/2, 0)
    return target

def conv_layer(num_qubits, param_prefix):
    qc = QuantumCircuit(num_qubits, name = 'Convolutional Layer')
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length = num_qubits*3)
    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc = qc.compose(
            conv_circuit(params[param_index:(param_index+3)]),
            [q1,q2]
        )
        qc.barrier()
        param_index += 3
    for q1, q2 in zip(qubits[1::2], qubits[2::2]+[0]):
        qc = qc.compose(
            conv_circuit(params[param_index:(param_index+3)]),
            [q1,q2]
        )
        qc.barrier()
        param_index += 3

    qc_inst = qc.to_instruction()
    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, qubits)
    return qc

def pool_circuit(params):
    target = QuantumCircuit(2)
    target.rz(-np.pi/2, 1)
    target.cx(1,0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0,1)
    target.ry(params[2], 1)
    return target

def pool_layer(sources, sinks, param_prefix):
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name = 'Pooling Layer')
    param_index = 0
    params = ParameterVector(param_prefix, length = num_qubits//2*3)
    for source, sink in zip(sources, sinks):
        qc = qc.compose(
            pool_circuit(params[param_index:(param_index+3)]),
            [source, sink]
        )
        qc.barrier()
        param_index += 3

    qc_inst = qc.to_instruction()
    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, range(num_qubits))
    return qc

quantum_instance = QuantumInstance(BasicAer.get_backend('qasm_simulator'), shots = 1024)
feature_map = ZFeatureMap(8)
ansatz = QuantumCircuit(8, name = 'Ansatz')
ansatz.compose(conv_layer(8, 'c1'), list(range(8)), inplace = True)
ansatz.compose(pool_layer([0,1,2,3], [4,5,6,7], 'p1'), list(range(8)), inplace = True)
ansatz.compose(conv_layer(4,'c2'), list(range(4,8)), inplace = True)
ansatz.compose(pool_layer([0,1],[2,3], 'p2'), list(range(4,8)), inplace = True)
ansatz.compose(conv_layer(2, 'c3'), list(range(6,8)), inplace = True)
ansatz.compose(pool_layer([0],[1],'p3'), list(range(6,8)), inplace = True)
circuit = QuantumCircuit(8)
circuit.compose(feature_map, range(8), inplace = True)
circuit.compose(ansatz, range(8), inplace = True)
observable = PauliSumOp.from_list(['Z'+'I'*7, 1])

qnn = TwoLayerQNN(
    num_qubits = 8,
    feature_map = feature_map,
    ansatz = ansatz,
    observable = observable,
    exp_val = AerPauliExpectation(),
    quantum_instance = quantum_instance
)

opflow_classifier = NeuralNetworkClassifier(
    qnn,
    optimizer = COBYLA(maxiter = 400),
    initial_point = algorithm_globals.random.random(qnn.num_weights)
)