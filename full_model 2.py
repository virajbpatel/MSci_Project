import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as col
import quanvolutional_filter as quanv
import QNN_Final3 as qnn

IMG_SIZE = 16
N_QUBITS_QUANV = 4
N_QUBITS_QNN = 4
N_QUBITS_QFC = 4
N_EPOCHS = 30
N_CIRCUITS = 8

class QuantumModel(torch.nn.Module):

    def __init__(self, skip) -> None:
        super().__init__()
        self.skip = skip
        self.qf1 = quanv.TrainableQuanvolutionalFilter(N_QUBITS_QUANV, IMG_SIZE, n_blocks = 5, pool = skip)
        self.k = np.sqrt(N_QUBITS_QUANV).astype(int)
        if skip:
            self.s = np.floor(IMG_SIZE/self.k).astype(int)
        else:
            self.s = np.floor((IMG_SIZE-1)/self.k).astype(int)
        self.qfc = quanv.QFC(N_QUBITS_QFC, input_dim = 16, encoding = '4x4_ryzxy')

    def forward(self, x, use_qiskit = False):
        x = x.view(-1, IMG_SIZE, IMG_SIZE)
        x = self.qf1(x, use_qiskit = use_qiskit)
        if not self.skip:
            x = F.avg_pool2d(x, self.k)
        x = F.max_pool2d(x, 2*self.k)
        x = x.reshape(-1, self.qfc.input_dim)
        x = self.qfc(x)
        print(x)
        return F.log_softmax(x, -1)

class HybridModel(torch.nn.Module):
    def __init__(self, N_QUBITS,N_CIRCUITS,N_BLOCKS,skip = True) -> None:
        super().__init__()
        self.nqubits = N_QUBITS
        self.ncircuits = N_CIRCUITS

        self.inputdim = 16
        self.circuit_depth = N_BLOCKS
        self.skip = skip
        self.qf1 = quanv.TrainableQuanvolutionalFilter(N_QUBITS_QUANV, IMG_SIZE, n_blocks = 5, pool = skip)
        self.k = np.sqrt(N_QUBITS_QUANV).astype(int)
        if skip:
            self.s = np.floor((IMG_SIZE)/self.k).astype(int)
        else:
            self.s = np.floor((IMG_SIZE-1)/self.k).astype(int)

        self.fc1 = torch.nn.Linear(N_QUBITS_QUANV*self.s*self.s, self.inputdim*self.ncircuits)
        self.qfc = quanv.QFC(self.nqubits, input_dim = self.inputdim, encoding = '4x4_ryzxy', n_blocks=self.circuit_depth)
        self.out = torch.nn.Linear(self.ncircuits*self.nqubits, 2)

    def forward(self, x, use_qiskit = False):
        x = x.view(-1, IMG_SIZE, IMG_SIZE)
        x = self.qf1(x, use_qiskit = use_qiskit)
        if not self.skip:
            x = F.avg_pool2d(x, self.k)#.view(N_QUBITS, self.s, self.s)
        x = x.reshape(-1, N_QUBITS_QUANV*self.s*self.s)

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
    

class full_model():
    def __init__(self, N_QUBITS, N_CIRCUITS, N_BLOCKS, LR, MOM, train_data, val_data, test_data):
        self.N_QUBITS = N_QUBITS
        self.N_CIRCUITS = N_CIRCUITS
        self.N_BLOCKS = N_BLOCKS
        self.LR = LR
        self.MOM = MOM
        self.trainset = train_data
        self.valset = val_data
        self.testset = test_data

        self.model = HybridModel(self.N_QUBITS, self.N_CIRCUITS, self.N_BLOCKS) 
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.LR,
                        momentum=self.MOM,nesterov=True)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=0.01, weight_decay=1e-4)
        self.loss = torch.nn.NLLLoss()
        self.loss_list_quantum = []

    def train(self,EPOCH):
        self.EPOCH = EPOCH
        self.model.train()
        self.accuracy_val=[]
        for epoch in range(self.EPOCH):
            total_loss = []
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
            print('Quantum Classifier performance, with {} perceptrons and {} layers of depth, on test data:\n\tLoss: {:.4f}\n\tAccuracy: {:.1f}%'.format(
                self.N_CIRCUITS,
                self.N_BLOCKS,
                self.loss_quantum,
                self.accuracy_quantum)
            )

    def save(self,iteration):
        torch.save(self.model,"Models/FULL_A{:.1f}_{}.pt".format(self.correct/len(self.testset)*100,iteration))
        np.savetxt("data/AccuracyListFULL_{}d_{}.csv".format(self.N_BLOCKS,iteration),self.accuracy_val)
        np.savetxt("data/LossListFULL_{}d_{}.csv".format(self.N_BLOCKS,iteration),self.loss_list_quantum)



#load in data sets
x_train = quanv.CustomImageDataset(
    annotations_file = 'brain_cancer_output/val.csv',
    img_dir = 'brain_cancer_output/val/Brain Tumor/',
    transform = transforms.Compose([transforms.ToPILImage(), transforms.Grayscale(), transforms.Resize(size = IMG_SIZE), transforms.ToTensor()])
)

x_val = quanv.CustomImageDataset(
    annotations_file = 'brain_cancer_output/test.csv',
    img_dir = 'brain_cancer_output/test/Brain Tumor/',
    transform = transforms.Compose([transforms.ToPILImage(), transforms.Grayscale(), transforms.Resize(size = IMG_SIZE), transforms.ToTensor()])
)

x_test = quanv.CustomImageDataset(
    annotations_file = 'brain_cancer_output/train.csv',
    img_dir = 'brain_cancer_output/train/Brain Tumor/',
    transform = transforms.Compose([transforms.ToPILImage(), transforms.Grayscale(), transforms.Resize(size = IMG_SIZE), transforms.ToTensor()])
)

train_loader = torch.utils.data.DataLoader(x_train, batch_size = 1, shuffle = True)
val_loader = torch.utils.data.DataLoader(x_val, batch_size = 1, shuffle = True)
test_loader = torch.utils.data.DataLoader(x_test, batch_size = 1, shuffle = True)


#neural network parameters
# lr=0.01
PERCEPTRON=8
CIRCUIT_DEPTH=8
lr=0.01
mom=0.1
epochs=10
run=6

print("Quantum Convolutional Neural Network Model with {} perceptrons, {} depth, {} epochs, {} iteration".format(PERCEPTRON,CIRCUIT_DEPTH,epochs,run))
quan_model = full_model(4,PERCEPTRON,CIRCUIT_DEPTH,lr,mom,train_loader,val_loader,test_loader)
quan_model.train(epochs)
quan_model.eval()
quan_model.save(run)
