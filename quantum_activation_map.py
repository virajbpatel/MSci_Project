import networkx as nx
import torchquantum as tq
from torchvision import transforms
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn.functional as F
from full_model import QuantumModel
import quanvolutional_filter as quanv
from tqdm import tqdm

IMG_SIZE = 18

def valid_test(dataloader, split, model, device, qiskit = False):
    target_all = []
    output_all = []
    with torch.no_grad():
        for data, label in tqdm(dataloader):
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

file = open('model12.pt', 'rb')

model = pickle.load(file)

x_test = quanv.CustomImageDataset(
    annotations_file = 'brain_cancer_output/train.csv',
    img_dir = 'brain_cancer_output/train/Brain Tumor/',
    transform = transforms.Compose([transforms.ToPILImage(), transforms.Grayscale(), transforms.Resize(size = IMG_SIZE), transforms.ToTensor()])
)

test_loader = torch.utils.data.DataLoader(x_test, batch_size = 1, shuffle = True)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

accuracy, loss = valid_test(test_loader, 'test', model, device, qiskit = False)
