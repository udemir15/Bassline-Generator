import os
import datetime as dt
import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from tqdm import tqdm

from metrics import calculate_accuracy, lehvenstein_distance

WANDB_API_KEY= '52c84ab3f3b5c1f999c7f5f389f5e423f46fc04a'
import wandb
wandb.login()


def main_wandb(model, criterion, optimizer, device, train_loader, validation_loader, params, project):

    N_epochs = params['training']['N_epochs']
    tf_ratio = params['training']['teacher_forcing_ratio']

    model_name = dt.datetime.strftime(dt.datetime.now(),"%d_%m__%H_%M")

    best_valid_loss = float('inf')

    with wandb.init(project=project, name=model_name, config=params, entity='nbg'):
        
        wandb.watch(model, log='all')
          
        samples = model.sample()
        print('\nBefore Training:')    
        print('\nSample:\n{}\n'.format(samples[0]))    
        wandb.log({'samples': samples})

        for epoch in tqdm(range(N_epochs)):

            train_loss, train_acc = train(model, train_loader, optimizer, criterion, device, tf_ratio)
            val_loss, val_acc, val_preds = test(model, validation_loader, criterion, device)
            
            # Track model predictions for debugging
            val_target, val_pred = torch.chunk(val_preds, 2, dim=0) 
            
            # Epoch dependent teacher forcing rate
            if epoch > int(N_epochs/4) and epoch < 3*int(N_epochs/4):
                tf_ratio -= 2*params['training']['teacher_forcing_ratio']/N_epochs
            
            wandb.log({'train_loss': train_loss, 'train_accuracy': train_acc,
                    'validation_loss': val_loss, 'validation_acc': val_acc,
                    'validation_targets': val_target, 'validation_preds': val_pred,
                    'teacher_forcing_ratio': tf_ratio})
            
            if val_loss < best_valid_loss:
                best_valid_loss = val_loss
                checkpoint(model_name, model, optimizer, epoch)

            if not (epoch % 25):
                print('Epoch: {}, train_loss: {:.6f}, val_loss: {:.6f}'.format(epoch, train_loss, val_loss))
                
            if not (epoch % 50):
                samples = model.sample()
                print('Sample:\n{}'.format(samples[0]))
                wandb.log({'samples': samples})
                
        samples = model.sample()
        print('\nAfter Training:') 
        print('\nSample:\n{}\n'.format(samples[0]))    
        wandb.log({'samples': samples})    


def main_simple(model, criterion, optimizer, device, train_loader, validation_loader, params):

    N_epochs = params['training']['N_epochs']
    tf_ratio = params['training']['teacher_forcing_ratio']

    samples = model.sample()
    print('\nBefore Training:')
    print('Initial Sample:\n{}\n'.format(samples[0]))    

    for epoch in range(N_epochs):
        
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device, tf_ratio)
        val_loss, val_acc, _ = test(model, validation_loader, criterion, device)
        
        if epoch > int(N_epochs/4) and epoch <= 3*int(N_epochs/4):
            tf_ratio -= 2*params['training']['teacher_forcing_ratio']/N_epochs
            if epoch == 3*int(N_epochs/4):
                tf_ratio = 0.0

        print('Epoch: {}, train_loss: {:.6f}, train_acc: {:.3f}, val_loss: {:.6f}, val_acc: {:.3f}'\
            .format(epoch, train_loss,train_acc, np.mean(val_loss), np.mean(val_acc)))
        
    samples = model.sample()
    print('\nAfter Training:')
    print('\nSample:\n{}\n'.format(samples[0]))    


def train(model, loader, optimizer, criterion, device, tf_ratio):
    """One epoch of training."""
    
    model.train()

    losses, accuracies  = [], []
    for source in loader:
        
        # source : (B, T+1)
        source = source.to(device)
        target = source.clone()

        optimizer.zero_grad()

        # activations shape: (B, K, T)
        activations = model(source, target, tf_ratio) 

        # SOS token is removed for loss and metric calculations
        target = target[:, 1:] # (B, T)
        
        loss = criterion(activations, target) 
        loss.backward()
        
        optimizer.step()
        loss.detach()
        losses.append(loss.item())
        
        y_pred = activations.argmax(1) # (B, T)

        # Metrics
        target, y_pred = target.cpu().numpy(), y_pred.cpu().numpy() 
        accuracies.append(calculate_accuracy(target, y_pred))
        #distances.append(lehvenstein_distance(target, y_pred))
        
    return np.mean(losses), np.mean(accuracies) #, np.mean(distances)

def test(model, loader, criterion, device):
    
    model.eval()
    
    losses, accuracies,  = [], []
    for source in loader:
        with torch.no_grad():
            
            # source, target shapes: (B, T+1)
            source = source.to(device)
            target = source.clone()

            # activations shape: (B, K, T)
            activations = model(source, target, 0.0) # turn off the teacher forcing ratio

            # SOS token is removed for loss and metric calculations
            target = target[:, 1:]  # (B, T)

            loss = criterion(activations, target) 
            losses.append(loss.item())  
            
            y_pred = activations.argmax(1)

            # return the target and the predictions for debugging
            visualization = torch.cat([target, y_pred], dim=0)

            # Metrics
            target, y_pred = target.cpu().numpy(), y_pred.cpu().numpy() 
            accuracies.append(calculate_accuracy(target, y_pred))
            #distances.append(lehvenstein_distance(target, y_pred))
            
    return np.mean(losses), np.mean(accuracies),  visualization


def checkpoint(model_name, model, optimizer, epoch):

    model_path = os.path.join('model_checkpoints', model_name+'.pt')

    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, model_path)


#grad_dict = {n: {'grads': [], 'ave': [], 'max': [], 'min': []}  for n, p in model.named_parameters() if ((p.requires_grad) and ("bias" not in n))}
#print_gradients(grad_dict)
#plot_grad_flow(grad_dict)
#track_gradients(model, grad_dict)

def track_gradients(model, grad_dict):
    for n, p in model.named_parameters():
        if (p.requires_grad) and ("bias" not in n):
            grad = p.grad.abs().cpu().numpy()
            grad_dict[n]['grads'].append(grad)
            grad_dict[n]['ave'].append(grad.mean())
            grad_dict[n]['max'].append(grad.max())
            grad_dict[n]['min'].append(grad.min())

def print_gradients(grad_dict):
    print('\n')
    for n, dct in grad_dict.items():
        print('{}'.format(n))
        print('max: {:.10f}, ave: {:.14f}, min: {:.18f}'.format(np.max(dct['max']), np.mean(dct['ave']), np.min(dct['min'])))
        grads = list(dct['grads'])
        unique_grads = np.unique(grads)
        print('{} | {}'.format(unique_grads[:3], unique_grads[-3:]))
   
#def plot_grad_flow(ave_grads, max_grads, layers): #named_parameters
def plot_grad_flow(grad_dict):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads, max_grads, min_grads = [], [], []
    for layer_dict in grad_dict.values():
        ave_grads.append(np.mean(layer_dict['ave']))
        min_grads.append(np.min(layer_dict['min']))
        max_grads.append(np.max(layer_dict['max']))
    print(min_grads)
    fig, ax = plt.subplots(figsize=(10,5))
    #ax.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    ax.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b", label='mean-gradient')
    ax.bar(np.arange(len(max_grads)), min_grads, alpha=0.1, lw=3, color="r", label='min-gradient')
    ax.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    ax.set_xticks(range(len(ave_grads)))
    ax.set_xticklabels(list(grad_dict.keys()),rotation="vertical")
    ax.set_xlim(left=0, right=len(ave_grads))
    ax.set_ylim(bottom = -0.00001, top=max(ave_grads)) # zoom in on the lower gradient regions
    ax.set_title("Gradient flow")
    ax.grid(True)
    ax.legend(loc=1)
    plt.show()
