import os

import numpy as np
import torch

import datetime as dt
from tqdm import tqdm
#import wandb
#wandb.login()

from dataloaders import load_data, create_loaders


#TODO: main_WANDB
def main(model, train_loader, test_loader, optimizer, criterion, train_args, device):
    
    train_losses, test_losses = [], [test(model, test_loader, criterion, device)]
    print('Test Loss Before Training: {:.6f}'.format(test_losses[-1]))
    
    for epoch in tqdm(range(train_args['N_epochs'])):
                      
        train_losses.append(train(model, train_loader, optimizer, criterion, device))
        
        #if epoch+1 % 5:
        print('Epoch: {}, train_loss:{:.6f}'.format(epoch+1, train_losses[-1]))
    
    test_losses.append(test(model, test_loader, criterion, device))
    print('Test Loss After Training: {:.6f}'.format(test_losses[-1]))
    
    #evaluate_classification(model, test_loader) # Print Metrics
    
    return train_losses, test_losses

def train(model, loader, optimizer, criterion, device):
    
    model.train()
    
    train_loss  = []
    for x in loader:
        
        x= x.to(device)  
                
        optimizer.zero_grad()
        y_pred = model(x)
        
        loss = criterion(y_pred, x)
        loss.backward() # retain_graph=True
        optimizer.step()
        
        train_loss.append(loss.item())
        loss.detach()
        
    return np.mean(train_loss)

def test(model, loader, criterion, device):
    
    model.eval()
    
    test_loss = []
    for x in loader:
        with torch.no_grad():
                        
            x = x.to(device)       
            y_pred = model(x)

            loss = criterion(y_pred, x)

            test_loss.append(loss.item())
    
    return np.mean(test_loss)

def checkpoint(model_name, model, optimizer, epoch):

    model_path = os.path.join('model_checkpoints', model_name+'.pt')

    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                #'train_loss': train_loss,
                #'test_loss': test_loss
                }, model_path)