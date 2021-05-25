import torch
import torch.nn as nn

from models import LSTMnetwork
    
class StackedUnidirLSTMDecoder(nn.Module):
    
    def __init__(self,
                 input_size,
                 num_layers,
                 dropout,
                 batch_size,
                 sequence_length,
                 device):
        
        super().__init__()
        
        self.sequence_length = sequence_length
        
        # decoder outputs are fed as inputs
        self.rnn = LSTMnetwork(input_size, input_size, 1, num_layers, dropout, batch_size, device)
                
    def forward(self, x, hidden):
        """
        input is the last output of the encoder network. shape: (batch, feat)
        """
        
        # (Batch, 1, feat)
        y = x.unsqueeze(dim=1)
        
        outputs = []
        for i in range(self.sequence_length): # for each time step
            
            y, hidden = self.rnn(y, hidden)        
            outputs.append(y) # record the output           
               
        return torch.cat(outputs, dim=1)
    
    def init_hidden_cell_states(self, random=False):
        return self.rnn.init_hidden_cell_states(random)

class StackedUnidirLSTMDenseDecoder(nn.Module):
    """
    Stacked Unidirectional LSTM followed by a Dense Layer
    """
    
    def __init__(self,
                 input_size,
                 output_size,
                 num_layers,
                 dropout,
                 batch_size,
                 sequence_length,
                 device):
        
        super().__init__()
        
        self.batch_size = batch_size
        self.input_size = input_size
        
        self.rnn =  StackedUnidirLSTMDecoder(input_size, num_layers, dropout, batch_size, sequence_length, device)
        self.dense = nn.Linear(input_size, output_size)
        # ACTIVATION ?????????????*
        
        
    def forward(self, x, hidden):
        
        y = self.rnn(x, hidden)
        output = self.dense(y)
        
        return output
    
    def init_hidden_cell_states(self, random=False):
        return self.rnn.init_hidden_cell_states(random)


class StackedUnidirLSTMDecoderwithEmbedding(nn.Module):
    
    def __init__(self,
                num_embeddings, # K by definition (for sampling procedure)
                embedding_dim, # input to the LSTMnetwork
                hidden_size, # hidden size of the LSTMnetwork
                num_layers,
                dropout,
                batch_size,
                sequence_length,
                device):
        
        super().__init__()

        self.num_embeddings = num_embeddings
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.batch_size = batch_size
       
        # Embed the inputs
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

        # decoder outputs are fed as inputs
        self.rnn = LSTMnetwork(embedding_dim, hidden_size, 1, num_layers, dropout, batch_size, device)

        self.dense = nn.Linear(hidden_size, num_embeddings)
                
    def forward(self, x, hidden):
        """
        input is the last output of the encoder network. (batch, feat)
        """
        
        # (Batch, 1, Feat)
        y = x.unsqueeze(dim=1)
        
        outputs = []
        for _ in range(self.sequence_length): # for each time step

            class_idx = y.argmax(-1)
            x = self.embedding(class_idx)
            
            y, hidden = self.rnn(x, hidden)   

            y = self.dense(y)

            outputs.append(y) # record the output            
               
        return torch.cat(outputs, dim=1)