import torch
import torch.nn as nn
import random

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
        x:  (B, T, F)
        h,c : (L*D, B, H)
        out: (B, T, D*H)-- last layer
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
    
    def __init__(self, embedding, encoder, decoder):
        
        super().__init__()
        
        self.embedding = embedding
        self.encoder = encoder
        self.decoder = decoder
                   
    def forward(self, x):
        
        x_embed = self.embedding(x)
                
        y, (h, c) = self.encoder(x_embed)
        
        if self.decoder.teacher_forcing_ratio != 0.0:
            output_sequence = self.decoder(y, (h, c), x_embed)
        else: 
            output_sequence = self.decoder(y, (h, c), None) 
            
        return output_sequence.permute(0,2,1) # for loss calculation
        
    # DECODER INPUT????????
    def sample(self ,x):
        with torch.no_grad():
            x = torch.zeros((self.decoder.batch_size, self.decoder.input_size)).cuda()        
            (h, c) = self.decoder.init_hidden_cell_states(random=True)    
            sample = self.decoder(x, (h, c), None)        
        return sample.argmax(dim=-1)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, teacher_forcing_ratio):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        self.teacher_forcing_ratio=teacher_forcing_ratio
       
    def forward(self, source, target):
        """source, target shapes: (B, T+1) """      

        #last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(torch.fliplr(source))
        
        #first input to the decoder is the <sos> tokens
        input = target[:,0]
      
        outputs = []
        for t in range(1, target.shape[1]):
            
            output, hidden, cell = self.decoder(input, hidden, cell)
            
            outputs.append(output)
            
            pred = output.argmax(1)
                    
            if random.random() < self.teacher_forcing_ratio:
                input = target[:,t] # use actual next token as next input
            else:
                input = pred
        
        return torch.stack(outputs, dim=2) # (B, K, T)

    def sample(self, N=10, T=64):
        with torch.no_grad():
            shape = (self.decoder.n_layers, N, self.decoder.hidden_size)
            hidden, cell = torch.randn(shape, device=self.device), torch.randn(shape, device=self.device)
            input = torch.zeros(N, dtype=torch.int64,device=self.device)
            predictions = []
            for _ in range(T):
                output, hidden, cell = self.decoder(input, hidden, cell)
                pred = output.argmax(-1)
                predictions.append(pred)
                input=pred
        return torch.stack(predictions, dim=1)

    def update_teacher_forcing_ratio(self, step_size):
        self.teacher_forcing_ratio -= step_size


class Seq2SeqOuz(nn.Module):
    
    def __init__(self, encoder, decoder):
        
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
                   
    def forward(self, targets):
        """targets: int tensor  (B, F)"""
                      
        _, hidden_enc = self.encoder(targets)

        SOS_vector = targets[:,0] # shape: (B)
        # teacher forcing with targets during training
        output_sequence = self.decoder(SOS_vector, hidden_enc, targets) # shape: (B, K, T)
        
        return output_sequence # .permute(0,2,1) # for loss calculation
              
    def sample(self, SOS=0):
        with torch.no_grad():
            x = SOS*torch.ones((self.decoder.batch_size), dtype=torch.int64).cuda()
            hidden = self.decoder.init_hidden_cell_states(random=True)
            samples = self.decoder(x, hidden, None)        
        return samples.argmax(1)

    def init_weights(self):
        for param in self.parameters():
            nn.init.normal_(param.data, mean=0, std=0.01)

    
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
        
        with torch.no_grad():
            x = torch.zeros((self.decoder.batch_size, self.decoder.num_embeddings), device=self.device)
            x[note_idx] = 1 # for argmax
            h, c = self.decoder.rnn.init_hidden_cell_states(random=True)
            sample = self.decoder(x, (h, c))
        return sample.argmax(dim=-1)