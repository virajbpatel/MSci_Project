import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set()
sns.set_theme(style = 'whitegrid')
'''
# Random vs trainable 

r_acc = np.loadtxt('random_quanv_acc.txt')
r_loss = np.loadtxt('random_quanv_loss.txt')

t_acc = np.loadtxt('train_quanv_acc.txt')
t_loss = np.loadtxt('train_quanv_loss.txt')

plt.plot(r_acc, label = 'Random')
plt.plot(t_acc, label = 'Trainable')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()
plt.close()

plt.plot(r_loss, label = 'Random')
plt.plot(t_loss, label = 'Trainable')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
plt.close()

# Image quality for Quanv+Classical

sizes = [16, 20, 22, 24, 32]
q_acc = [85.790, 86.720, 86.454, 85.657, 84.728]
c_acc = [79.947, 85.657, 87.118, 87.915, 88.181]

plt.plot(sizes, c_acc, label = 'Classical')
plt.plot(sizes, q_acc, label = 'Quantum')
plt.legend()
#plt.title('Accuracy against image size for 4 qubits')
plt.xlabel('Image width in pixels')
plt.ylabel('Accuracy')
plt.show()
plt.close()

# QFC circuit depth for Quanv+QFC

n_layers = [4, 6, 8]
q_acc = [76.096, 72.908, 75.033]
plt.plot(n_layers, q_acc)
plt.title('Number of layers in QFC against accuracy')
plt.xlabel('Number of layers')
plt.ylabel('Accuracy')
plt.show()
plt.close()

# Quantum vs Classical plots

import csv

c_acc = np.loadtxt('classical_hybrid_acc6.txt')
c_loss = np.loadtxt('classical_hybrid_loss6.txt')
q_acc_x = []
q_acc_y = []
q_loss_x = []
q_loss_y = []

with open('acc1.csv', 'r') as file:
    lines = csv.reader(file, delimiter = ',')
    for row in lines:
        q_acc_x.append(float(row[0]))
        q_acc_y.append(float(row[1]))

with open('loss1.csv', 'r') as file:
    lines = csv.reader(file, delimiter = ',')
    for row in lines:
        q_loss_x.append(float(row[0]))
        q_loss_y.append(float(row[1]))

epochs = list(range(30))

plt.plot(epochs, c_acc, label = 'Classical')
plt.plot(q_acc_x, q_acc_y, label = 'Quantum')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.close()

plt.plot(epochs, c_loss, label = 'Classical')
plt.plot(q_loss_x, q_loss_y, label = 'Quantum')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.close()

'''
c_acc = np.loadtxt('classical_qfc_acc6.txt')
c_loss = np.loadtxt('classical_qfc_loss6.txt')
q_acc = np.loadtxt('quantum_model_acc8.txt')
q_loss = np.loadtxt('quantum_model_loss8.txt')

plt.plot(c_acc, label = 'Classical')
plt.plot(q_acc, label = 'Quantum')
#plt.title('QFC vs Classical')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.close()

plt.plot(c_loss, label = 'Classical')
plt.plot(q_loss, label = 'Quantum')
#plt.title('QFC vs Classical')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.close()

# Quanv circuit depth

layers = [1,2,3,4,5]
acc_16 = [85.525,87.251,85.923,85.790,84.993]
acc_20 = [86.985,86.985, 86.587, 85.790]
acc_24 = [86.056,86.056,86.321]

plt.plot(layers, acc_16, label = r'$16\times16$')
plt.plot(layers[:-1], acc_20, label = r'$20\times20$')
plt.plot(layers[:-2], acc_24, label = r'$24\times24$')
plt.legend()
plt.title('Effect of circuit depth in Quanvolution')
plt.xlabel('Number of Layers')
plt.ylabel('Accuracy')
plt.show()
plt.close()