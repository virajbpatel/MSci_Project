import matplotlib.pyplot as plt

sizes = [16, 20, 22, 24, 32]
q_acc = [85.790, 86.720, 86.454, 85.657, 84.728]
c_acc = [79.947, 85.657, 87.118, 87.915, 88.181]

plt.plot(sizes, c_acc, label = 'Classical')
plt.plot(sizes, q_acc, label = 'Quantum')
plt.legend()
plt.title('Accuracy against image size for 4 qubits')
plt.xlabel('Image width in pixels')
plt.ylabel('Accuracy')
plt.show()
plt.close()

n_layers = [3, 5]
q_acc = [85.259, 85.790]