import os
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

def load_data(data_params): #, dataset_prefix='bassline_representations'
    
    dataset_path, scale_type, M = data_params['dataset_path'], data_params['scale_type'], data_params['M']
    
    dataset_name = data_params['dataset_name'] +'_{}_M{}.csv'.format(scale_type, M)

    dataset_dir = os.path.join(dataset_path, dataset_name)

    df = pd.read_csv(dataset_dir, header=None)

    # First column is title
    X = df[df.columns[1:]].to_numpy()
    
    return X

def append_EOS(X):
    EOS_token = X.max()+1
    return np.concatenate( (X, EOS_token*np.ones((X.shape[0],1), dtype=np.int64)), axis=1)

def append_SOS(X, SOS_token=-1):
    X = np.concatenate( (SOS_token*np.ones((X.shape[0],1), dtype=np.int64), X), axis=1)    
    return X+1 
    
# DELETE
def fix_class_labels(X):
    
    X[ X!= 0] -= 25

    X[X==75] = 36
    
    return X

class DataSet(Dataset):
    
    def __init__(self, X):
        self.X = X
        
    def __getitem__(self, idx):
        return self.X[idx]
    
    def __len__(self):
        return len(self.X)
    
def create_loaders(X, data_params, train_ratio=0.75, validation_ratio=0.15, test_ratio=0.1):
    
    #X_train, X_test = train_test_split(X, test_size=test_size, random_state=42)
    x_train, x_test, = train_test_split(X, test_size=1 - train_ratio, random_state=42, shuffle=True)
    x_val, x_test  = train_test_split(x_test, test_size=test_ratio/(test_ratio + validation_ratio), random_state=42, shuffle=False)

    train_set, validation_set, test_set = DataSet(x_train), DataSet(x_val), DataSet(x_test)

    train_loader = DataLoader(train_set, batch_size=data_params['batch_size'], shuffle=True, drop_last=True)
    validation_loader = DataLoader(validation_set, batch_size=data_params['batch_size'], shuffle=True, drop_last=True)    
    test_loader = DataLoader(test_set, batch_size=data_params['batch_size'], drop_last=True)
    
    return train_loader, validation_loader, test_loader 
    

