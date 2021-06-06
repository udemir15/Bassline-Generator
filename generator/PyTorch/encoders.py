import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers):
        
        super().__init__()
        
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, n_layers, batch_first=True)
        
        self.init_weights()

    def forward(self, src):        
        """src: shape (B, T+1)"""
        
        # embedded: shape (T+1, B, E)     
        embedded = self.embedding(src)

        # hidden, cell shapes: (L * D, B, H)
        _, (hidden, cell) = self.rnn(embedded)
        
        return hidden, cell

    def init_weights(self):
        #nn.init.kaiming_uniform_(self.embed_out, a=math.sqrt(5))
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)


class GRUEncoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers):
        
        super().__init__()
        
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.GRU(embedding_size, hidden_size, n_layers, batch_first=True)
        
        self.init_weights()

    def forward(self, src):        
        """src: shape (B, T+1)"""
        
        embedded = self.embedding(src) # (B, T+1, E)

        # hidden shape: (L*D, B, H)
        _, hidden = self.rnn(embedded)
        
        return hidden

    def init_weights(self):
        #nn.init.kaiming_uniform_(self.embed_out, a=math.sqrt(5))
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)


class BidirectionalGRUEncoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size):
        
        super().__init__()
        
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.GRU(embedding_size, hidden_size, 1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, output_size)

        self.init_weights()

    def forward(self, src):        
        """ src: shape (B, T+1)
            outputs: shape (B, T+1, H*D)
            hidden: shape (B, O)
        """
        
        embedded = self.embedding(src) # (B, T+1, E)

        # hidden, shape: (L*D, B, H)
        # outputs shape: (B, T+1, H*2)
        outputs, hidden = self.rnn(embedded)

        #final_hidden_forward = hidden[-2, :, :]
        #final_hidden_backward = hidden[-1, :, :]
        cat_hiddens = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1) # (B, 2*H)

        hidden = torch.tanh(self.fc(cat_hiddens)) # (B, O)

        return outputs, hidden

    def init_weights(self):
        #nn.init.kaiming_uniform_(self.embed_out, a=math.sqrt(5))
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)