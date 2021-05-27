import torch
import torch.nn as nn

from models import LSTMnetwork

class StackedUnidirLSTMEncoder(nn.Module):
    """
    Stacked Unidirectional LSTM Encoder
    """
    
    def __init__(self,
                input_size, #embedding_dim
                hidden_size,
                num_layers,
                dropout,
                batch_size,
                device):
        
        super().__init__()
        
        self.net = LSTMnetwork(input_size, hidden_size, 1, num_layers, dropout, batch_size, device).to(device)
        
    def forward(self, x):             
        """Takes embedded inputs x: shape (Batch, Time, Embed)"""
        
        y, (h, c) = self.net(x)
            
        return y[:,-1,:], (h, c)
    
    
class StackedBidirLSTMEncoder(nn.Module):
    
    def __init__(self,
                 input_size,
                hidden_size,
                num_layers,
                dropout,
                batch_size,
                device):
        
        super().__init__()
        
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        
        self.net = LSTMnetwork(input_size, hidden_size, 2, num_layers, dropout, batch_size, device).to(device)
        
    def forward(self, x):
        """Takes embedded inputs x: shape (Batch, Time, Embed)"""
        
        y, (h, c) = self.net(x)
        
        # reshape to (Batch, Seq, Directions, Hidden) and sum both directions
        y = torch.sum(y.reshape(self.batch_size, -1, 2, self.hidden_size), dim=2)
        
        # reshape to (Layers, Directions, Batch, Hidden) and sum both directions
        h = torch.sum(h.reshape(self.num_layers, 2, self.batch_size, self.hidden_size), dim=1)
        c = torch.sum(c.reshape(self.num_layers, 2, self.batch_size, self.hidden_size), dim=1)
                
        # return last time step of the output
        return y[:,-1,:], (h, c)


class Seq2SeqEncoder(nn.Module):
    
    def __init__(self,
                num_embeddings, # number of embeddings
                embedding_size,                
                hidden_size,
                num_layers,
                dropout,
                batch_size,
                device):
        
        super().__init__()
        
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(num_embeddings, embedding_size)
        self.net = LSTMnetwork(embedding_size, hidden_size, 2, num_layers, dropout, batch_size, device).to(device)
        
    def forward(self, x):
        """Takes class labels x: shape (Batch, Time)"""

        x = self.embedding(x) # shape: (B, T, E)
        
        y, (h, c) = self.net(x)
        
        # reshape to (Batch, Seq, Directions, Hidden) and sum both directions
        y = torch.sum(y.reshape(self.batch_size, -1, 2, self.hidden_size), dim=2)
        
        # reshape to (Layers, Directions, Batch, Hidden) and sum both directions
        h = torch.sum(h.reshape(self.num_layers, 2, self.batch_size, self.hidden_size), dim=1)
        c = torch.sum(c.reshape(self.num_layers, 2, self.batch_size, self.hidden_size), dim=1)
                
        # return last time step of the output
        return y[:,-1,:], (h, c)