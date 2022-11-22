import random
import os
import pandas as pd

dir = "brain_cancer_output/train/Brain Tumor/"
file_list = random.sample(os.listdir(dir), 50)

df = pd.read_csv('brain_cancer_output/train.csv')
df = df[df['Image'].isin(file_list)]
class_list = df['Class'].to_list()
print(class_list)