import torch
# import torch.utils.data as data
import os
import pandas as pd
import numpy as np

class BasicDataset(object):
    def __init__(self, root_dir, train = True):
        super(BasicDataset, self).__init__()

        self.root_dir = root_dir
        self.train = train
        
        if train is True:
            self.x_dir = os.path.join(root_dir, 'train_feature.csv')
            self.y_dir = os.path.join(root_dir, 'train_label.csv')
            
            self.x_file = pd.read_csv(self.x_dir)
            self.y_file = pd.read_csv(self.y_dir)
        
        else:
            self.x_dir = os.path.join(root_dir, 'test_feature.csv')
            self.y_dir = os.path.join(root_dir, 'test_label.csv')
            
            self.x_file = pd.read_csv(self.x_dir)
            self.y_file = pd.read_csv(self.y_dir)
            

            
    def __getitem__(self, index):
        
        x = np.array(self.x_file.iloc[index], dtype=np.double)
        y = int(self.y_file.iloc[index])

        return x, y

    def __len__(self):
        return len(self.x_file)
        
if __name__ == "__main__":
    train_dataset = BasicDataset(root_dir='./dataset', train = True)
    print (train_dataset.__getitem__(0))
    print (train_dataset[0])