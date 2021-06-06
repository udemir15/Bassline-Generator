import torch
import torch.nn as nn


class Seq2SeqGRU(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
       
    def forward(self, source, target, teacher_forcing_ratio):
        """source, target shapes: (B, T+1) """      

        #last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden = self.encoder(source)

        outputs = self.decoder(target, hidden, hidden, teacher_forcing_ratio) # context is the hidden state
        
        return outputs

    def sample(self, N=10, T=64):

        with torch.no_grad():
            # Initial hidden states
            hidden = self.decoder.init_hidden(N)

            # zero input, <SOS> token, shape (N, T+1) to accomodate the decoder unrolling
            input = torch.zeros((N,T+1), dtype=torch.int64, device=self.device)

            output = self.decoder(input, hidden, hidden, 0.0) # turn off the teacher forcing ratio

            # Get the most probable tokens
            predictions = output.argmax(1)
        
        return predictions


class Seq2SeqGRUWithAttention(nn.Module):
    def __init__(self, encoder, decoder, device):

        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
       
    def forward(self, source, target, teacher_forcing_ratio):
        """
        source, target shapes: (B, T+1)
        outputs: (B, E, T)
        """      

        #last hidden state of the encoder is used as the initial hidden state of the decoder
        _, hidden = self.encoder(source)

        outputs, _ = self.decoder(target, hidden, teacher_forcing_ratio) # context is the hidden state

        return outputs

    # TODO ONLY DECODER, INIT HIDDEN ??
    def sample(self, note=14, N=10, T=64):

        with torch.no_grad():

            trg = torch.zeros((N, 1), dtype=torch.int64, device=self.device)

            trg = torch.cat((trg, note*torch.ones((N,T), dtype=torch.int64, device=self.device)), dim=1)

            output = self(trg, trg, 0.0)

            # Get the most probable tokens
            predictions = output.argmax(1)
        
        return predictions

    def reconstruct_bassline(self, bassline):
        
        with torch.no_grad():
            
            _, hidden = self.encoder(bassline)
            
            outputs, attentions = self.decoder(bassline, hidden, 0) # context is the hidden state
            
            reconstruction = outputs.argmax(1)
            
        return reconstruction, attentions[:,:,:-1]


class Seq2SeqLSTMSimple(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
       
    def forward(self, source, target, teacher_forcing_ratio):
        """source, target shapes: (B, T+1) """      

        #last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(torch.fliplr(source))

        outputs = self.decoder(target, hidden, cell, teacher_forcing_ratio)
        
        return outputs

    def sample(self, N=10, T=65):

        with torch.no_grad():
            # Initial hidden, cell states
            hidden, cell = self.decoder.init_hidden_cell(N)

            # zero input, <SOS> token, shape (N, T) to accomodate the decoder unrolling
            input = torch.zeros((N,T), dtype=torch.int64, device=self.device)

            output = self.decoder(input, hidden, cell, 0.0)
            predictions = output.argmax(1)
        
        return predictions


class Seq2SeqGRUSimple(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
       
    def forward(self, source, target, teacher_forcing_ratio):
        """source, target shapes: (B, T+1) """      

        #last hidden state of the encoder is used as the initial hidden state of the decoder

        source = torch.fliplr(source)

        hidden = self.encoder(source)

        outputs = self.decoder(target, hidden, teacher_forcing_ratio)
        
        return outputs

    def sample(self, N=10, T=65):

        with torch.no_grad():
            # Initial hidden states
            hidden = self.decoder.init_hidden(N)

            # zero input, <SOS> token, shape (N, T) to accomodate the decoder unrolling
            input = torch.zeros((N,T), dtype=torch.int64, device=self.device)

            output = self.decoder(input, hidden, 0.0)
            predictions = output.argmax(1)
        
        return predictions