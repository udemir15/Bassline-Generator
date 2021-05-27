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
                device,
                teacher_forcing_ratio=0.0):
        
        super().__init__()
        
        self.sequence_length = sequence_length
        self.teacher_forcing_ratio=teacher_forcing_ratio
        
        self.rnn = LSTMnetwork(input_size, input_size, 1, num_layers, dropout, batch_size, device)
                
    def forward(self, y, hidden, targets=None):
        """
        y (tensor): (batch, feat)
        hidden (tensor): 
        targets (tensor, default=None): embedded bassline sequence for teacher forcing. shape: (Batch, Time, Embed)
        """
                
        y = y.unsqueeze(dim=1) # (Batch, 1, feat)       
        
        outputs = []        
        if targets is None:
                    
            for _ in range(self.sequence_length): # for each time step
                y, hidden = self.rnn(y, hidden)        
                outputs.append(y) # record the output  
                
        else: # Teacher Forcing
    
            for i in range(self.sequence_length): # for each time step
                
                if self.teacher_forcing_ratio > torch.rand(1):

                    # targets: shape (Batch, Time, Embed) 
                    y = targets[:,i,:].unsqueeze(dim=1) 
                    
                y, hidden = self.rnn(y, hidden)        
                outputs.append(y) # record the output
                
        return torch.cat(outputs, dim=1)
                        
    def init_hidden_cell_states(self, random=False):
        return self.rnn.init_hidden_cell_states(random)
    
    def update_teacher_forcing_ratio(self, step_size, epoch, total_epochs):        
        if self.teacher_forcing_ratio:
            self.teacher_forcing_ratio -= step_size*(epoch/total_epochs)
        

    
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
                device,
                teacher_forcing_ratio=0.0):
        
        super().__init__()
        
        self.batch_size = batch_size
        self.input_size = input_size
        self.teacher_forcing_ratio = teacher_forcing_ratio
        
        self.rnn =  StackedUnidirLSTMDecoder(input_size, num_layers, dropout, batch_size, sequence_length, device, teacher_forcing_ratio)
        
        # ACTIVATION ?????????????*******************
        self.dense = nn.Linear(input_size, output_size)
        
    def forward(self, x, hidden, targets=None):        
        y = self.rnn(x, hidden, targets)
        output = self.dense(y)        
        return output
    
    def init_hidden_cell_states(self, random=False):
        return self.rnn.init_hidden_cell_states(random)
    
    def update_teacher_forcing_ratio(self, step_size, epoch, total_epochs):        
        self.rnn.update_teacher_forcing_ratio(step_size, epoch, total_epochs)
        self.teacher_forcing_ratio = self.rnn.teacher_forcing_ratio        


class Seq2SeqDecoder(nn.Module):
    
    def __init__(self,
                num_embeddings, # number of embeddings
                embedding_size,
                hidden_size, # must be equal to hidden_enc
                num_layers,
                dropout,
                batch_size,
                sequence_length,
                device,
                teacher_forcing_ratio=0.0):
        
        super().__init__()
        
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.teacher_forcing_ratio=teacher_forcing_ratio
         
        self.embedding = nn.Embedding(num_embeddings, embedding_size)
        self.rnn = LSTMnetwork(embedding_size+hidden_size, hidden_size, 1, num_layers, dropout, batch_size, device)
        self.linear = nn.Linear(2*hidden_size+embedding_size, num_embeddings) # reduce to number of clasees for argmax
    
    def forward(self, y0, hidden_enc, targets):
        """
        y0: (tensor shape: (Batch)) SOS token 
        z: (tensor, shape: (1, B, H) ): latent
        hidden_enc (tupple of tensors shapes: (1, B, H)) : 
        targets (tensor shape: (Batch, Time)): bassline sequence for teacher forcing. 
        """
                        
        y = y0 # first input is the SOS token, shape: (Batch)
        hidden = hidden_enc # first decoder hidden state is the last hidden state of the encoder

        z = hidden_enc[0].squeeze(dim=0) # latent is the last hidden state of the encoder

        outputs = []
        if targets is None:

            for i in range(self.sequence_length): # for each time step

                y_embed = self.embedding(y) # (B, E)

                # the latent is concat at each step
                y = torch.cat((z, y_embed), dim=1).unsqueeze(dim=1) # shape: (B, T=1, H+E)   

                rnn_out, hidden = self.rnn(y, hidden) # shape: (B, T=1, H+E)

                rnn_out = torch.cat((z, y_embed, rnn_out.squeeze(dim=1)), dim=1) # (B, H+E+H)
                y = self.linear(rnn_out) # (B, Out)     
                outputs.append(y) # record the output tensor for loss calculation  
                
                y = y.argmax(-1) # most probable token for next step 
        else:

            for i in range(self.sequence_length): # for each time step           

                if self.teacher_forcing_ratio > torch.rand(1):                                   
                    y = targets[:,i] # (Batch)

                y_embed = self.embedding(y) # (B, E)

                # the latent is concat at each step
                y = torch.cat((z, y_embed), dim=1).unsqueeze(dim=1) # shape: (B, T=1, H+E)   

                rnn_out, hidden = self.rnn(y, hidden) # shape: (B, T=1, H+E)

                rnn_out = torch.cat((z, y_embed, rnn_out.squeeze(dim=1)), dim=1) # (B, H+E+H)
                y = self.linear(rnn_out) # (B, Out)     
                outputs.append(y) # record the output tensor for loss calculation  
                
                y = y.argmax(-1) # most probable token for next step 
        
        return torch.stack(outputs, dim=2) # (B, K, T)
                        
    def init_hidden_cell_states(self, random=False):
        return self.rnn.init_hidden_cell_states(random)
    
    def update_teacher_forcing_ratio(self, step_size):
        self.teacher_forcing_ratio -= step_size

        
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