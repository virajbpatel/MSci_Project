import pennylane as qml
import numpy as np
from pennylane.templates import RandomLayers
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tqdm import tqdm

n_epochs = 30
n_layers = 1
n_train = 50
n_test = 30

SAVE_PATH = 'quanvolution/'
PREPROCESS = True
np.random.seed(0)
tf.random.set_seed(0)

mnist_dataset = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist_dataset.load_data()

train_images = train_images[:n_train]
train_labels = train_labels[:n_train]
test_images = test_images[:n_test]
test_labels = test_labels[:n_test]

train_images = train_images/255
test_images = test_images/255

#print(train_images[0].flatten())

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

