import os

import numpy as np
import torch

import datetime as dt
from tqdm import tqdm
#import wandb
#wandb.login()

from sklearn.metrics import accuracy_score

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


def calculate_accuracy(target_batch, input_batch):
    """shapes: (B, T) """
       
    accuracies = [accuracy_score(t, i, normalize=True) for t, i in zip(target_batch, input_batch)]
       
    #for i, acc in enumerate(accuracies):
        #if acc > 0.9:
        #    print('Target')
        #    print(target_batch[i,:])
        #    print('Input')
        #    print(input_batch[i,:])
        #    print('\n')
      
    return np.sum(accuracies) / target_batch.shape[0]


def train(model, loader, optimizer, criterion, device):
    """One epoch of training."""
    
    model.train()
    
    train_losses, batch_accuracies  = [], []
    for x in loader:
        
        x = x.to(device) # shape: (B, T)
                
        optimizer.zero_grad()
        activations = model(x) #shape: (B, K, T)
        
        loss = criterion(activations, x)
        loss.backward()
        optimizer.step()
        
        y_pred = activations.argmax(1)
        acc = calculate_accuracy(x.cpu().numpy(), y_pred.cpu().numpy())
        batch_accuracies.append(acc)
        
        loss.detach()
        train_losses.append(loss.item())        
        
    mean_epoch_loss = np.mean(train_losses)
    mean_epoch_accuracy = np.mean(batch_accuracies)
        
    return mean_epoch_loss, mean_epoch_accuracy

def test(model, loader, criterion, device):
    
    model.eval()
    
    test_losses, batch_accuracies = [], []
    for x in loader:
        with torch.no_grad():
                        
            x = x.to(device)       
            activations = model(x)

            loss = criterion(activations, x)
            test_losses.append(loss.item())
            
            y_pred = activations.argmax(1)
            acc = calculate_accuracy(x.cpu().numpy(), y_pred.cpu().numpy())
            batch_accuracies.append(acc)
            
    mean_test_loss = np.mean(test_losses)
    mean_test_accuracy = np.mean(batch_accuracies) 
    
    return mean_test_loss, mean_test_accuracy


def checkpoint(model_name, model, optimizer, epoch):

    model_path = os.path.join('model_checkpoints', model_name+'.pt')

    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, model_path)