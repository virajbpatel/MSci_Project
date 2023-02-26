#Need to figure out how to import things all at once
from QNN_Final import *
import pandas as pd
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

#This part is only needed for my Mac
import os 
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#load in data sets
x_train = CustomImageDataset(
    annotations_file = 'brain_cancer_output/val.csv',
    img_dir = 'brain_cancer_output/val/Brain Tumor/',
    transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
)

x_val = CustomImageDataset(
    annotations_file = 'brain_cancer_output/test.csv',
    img_dir = 'brain_cancer_output/test/Brain Tumor/',
    transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
)

x_test = CustomImageDataset(
    annotations_file = 'brain_cancer_output/train.csv',
    img_dir = 'brain_cancer_output/train/Brain Tumor/',
    transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
)

train_loader = torch.utils.data.DataLoader(x_train, batch_size = 1, shuffle = True)
val_loader = torch.utils.data.DataLoader(x_val, batch_size = 1, shuffle = True)
test_loader = torch.utils.data.DataLoader(x_test, batch_size = 1, shuffle = True)

#neural network parameters
PERCEPTRONS=8
lr=0.002
epochs=50

dataset_1=[]
for i in range(1,5):
    print("Quantum Model with {} perceptrons, {} learning rate, {} epochs, {} iteration".format(PERCEPTRONS,lr,epochs,i+1))
    quan_model = quantum_model(4,PERCEPTRONS,lr,0.1,train_loader,val_loader,test_loader)
    quan_model.train(epochs)
    quan_model.eval()
    data_i=quan_model.save(i+1)

#data saving things
circuits=[]
learning_rates=[]
epochs=[]
quantum_loss=[]
tp_q=[]
tn_q=[]
quantum_accuracy=[]
iteration=[]

data_set=[circuits,
    learning_rates,
    epochs,
    quantum_loss,
    tp_q,
    tn_q,
    quantum_accuracy,
    iteration
    ]

for j in range(len(data_set)):
    for i in range(len(dataset_1)):
        data_set[j].append(dataset_1[i][j])
        
dict={"Perceptrons":data_set[0],
    "Learning Rate":data_set[1],
    "Epochs":data_set[2],
    "Loss":data_set[3],
    "Correct Positive":data_set[4],
    "Correct Negative":data_set[5],
    "Quantum Accuracy":data_set[6],
    "Iteration":data_set[7]
    }

df = pd.DataFrame(dict) 
df.to_csv('data/Quantum_Dataset_p{}_e{}_lr{}.csv'.format(8,50,0.002)) 