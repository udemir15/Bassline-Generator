import torch
import torch.nn as nn


class LSTMnetwork(nn.Module):
    
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_directions,
                 num_layers,
                 dropout,
                 batch_size,
                 device):
        
        super().__init__()
        
        self.num_directions = num_directions
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        bidirectional = True if num_directions==2 else False
        self.device = device
        
        self.lstm = nn.LSTM(input_size,
                            hidden_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            bidirectional=bidirectional,
                            batch_first=True)
                       
    def forward(self, x, hidden=None):
        """
        Processes all the time steps in the batch at once.
        """
        
        if hidden is None:
            hidden = self.init_hidden_cell_states(random=False)
        
        lstm_out, (hn, cn) = self.lstm(x, hidden)
        
        return lstm_out, (hn, cn)
    
    def init_hidden_cell_states(self, random=False):

        shape = (self.num_layers*self.num_directions, self.batch_size, self.hidden_size)
        if random:
            return (torch.randn(shape, device=self.device), torch.randn(shape, device=self.device))                     
        else:
            return (torch.zeros(shape, device=self.device), torch.zeros(shape, device=self.device))
    
        
class VanillaAutoEncoder(nn.Module):
    
    def __init__(self, encoder, decoder):
        
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
                   
    def forward(self, x):
                
        y, (h, c) = self.encoder(x)
        output_sequence = self.decoder(y, (h, c))
               
        return output_sequence.permute(0,2,1) # for loss calculation
    
    def sample(self ,x):
        
        x = torch.zeros((self.decoder.batch_size, self.decoder.input_size)).cuda()
        
        (h, c) = self.decoder.init_hidden_cell_states(random=True)
        
        sample = self.decoder(x, (h, c))
        
        return sample.argmax(dim=-1)
    

class IOTransformer(nn.Module):    
    
    def __init__(self, encoder_hidden_size, decoder_hidden_size, decoder_num_embeddings):
        
        super().__init__()
            
        self.input_transformer = nn.Linear(encoder_hidden_size, decoder_num_embeddings)
        self.hidden_transformer = nn.Linear(encoder_hidden_size, decoder_hidden_size)
        self.cell_transformer = nn.Linear(encoder_hidden_size, decoder_hidden_size)
   
    def forward(self, x, h, c):
        
        # Match the inputs via linear transformations        
        y = self.input_transformer(x)
        h = self.hidden_transformer(h)
        c = self.cell_transformer(c) 
        
        return y, (h, c)
    
    
class AutoEncoder(nn.Module):

    def __init__(self, encoder, decoder, device):

        super().__init__()

        self.device = device
        
        self.encoder = encoder
        # Linear layers for transforming input shapes
        self.transformer = IOTransformer(encoder.hidden_size, decoder.hidden_size, decoder.num_embeddings)        
        self.decoder = decoder
        

    def forward(self, x):

        y, (h, c) = self.encoder(x)
        
        # Match the inputs via linear transformations      
        y, (h, c) = self.transformer(y, h, c)

        output_sequence = self.decoder(y, (h, c))

        return output_sequence.permute(0,2,1) # for loss calculation

    def sample(self, note_idx):
        """
        Feeds the decoder with a tensor corresponding to the given note_idx to sample batch of basslines.
        """
        x = torch.zeros((self.decoder.batch_size, self.decoder.num_embeddings), device=self.device)
        x[note_idx] = 1 # for argmax

        h, c = self.decoder.rnn.init_hidden_cell_states(random=True)

        sample = self.decoder(x, (h, c))

        return sample.argmax(dim=-1)