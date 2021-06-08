import os, json
import datetime as dt

import numpy as np

import torch
import torch.nn as nn

import models
import encoders
import decoders
from training import main_wandb 
from dataloaders import load_data, make_loaders, append_SOS

SEED = 27

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

PROJECT_NAME = 'seq2seq_gru_simple'

if __name__ == '__main__':
    
    with open('parameters/Seq2SeqGRUSimple_params.json', 'r') as infile: # load params
        params = json.load(infile)

    data_params = params['data']
    train_params = params['training']
    
    X, titles = load_data(data_params)
    X = append_SOS(X)

    K = X.max()+1 # Number of classes, assumes consecutive [0,max] inclusive
    sequence_length = X.shape[1]
    
    print('Number of classes: {}\nSequence Length: {}'.format(K, sequence_length))
    print('Number of data points: {}'.format(X.shape[0]))

    encoder_params = params['encoder']
    encoder_params['input_size'] = K
    decoder_params = {'output_size': K,
                    'embedding_size': encoder_params['embedding_size'],
                    'hidden_size': encoder_params['hidden_size'],
                    'n_layers': encoder_params['n_layers']}

    train_loader, test_loader = make_loaders(X, train_params['batch_size'])

    encoder = encoders.GRUEncoder(**encoder_params)
    decoder = decoders.SimpleGRUDecoder(**decoder_params)

    model = models.Seq2SeqGRUSimple(encoder, decoder, device).to(device)
    print(model)
    print('Number of parameters: {}\n'.format(sum([parameter.numel() for parameter in model.parameters()])))

    criterion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=train_params['lr'])

    main_wandb(model, criterion, optimizer, device, train_loader, test_loader, params, PROJECT_NAME)

    


