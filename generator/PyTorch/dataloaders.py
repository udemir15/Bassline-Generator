import os
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

def load_data(data_params):
    
    dataset_path, scale_type, M = data_params['dataset_path'], data_params['scale_type'], data_params['M']
    
    dataset_name = 'bassline_representations_{}_M{}.csv'.format(scale_type, M)

    dataset_dir = os.path.join(dataset_path, dataset_name)

    df = pd.read_csv(dataset_dir, header=None)

    # First column is title
    X = df[df.columns[1:]].to_numpy()
    
    X = fix_class_labels(X)
    
    return X
    
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
    
def create_loaders(X, data_params, test_size=0.2):
    
    X_train, X_test = train_test_split(X, test_size=test_size, random_state=42)

    train_set, test_set = DataSet(X_train), DataSet(X_test)

    train_loader = DataLoader(train_set, batch_size=data_params['batch_size'], shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=data_params['batch_size'], drop_last=True)
    
    return train_loader, test_loader 
    