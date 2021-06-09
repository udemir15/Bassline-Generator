import torch
import torch.nn as nn
import torch.nn.functional as F

import random


class GRUDecoder(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size):
        """Since this is an AutoEncoder Seq2Seq hybrid, the input size is the output size."""

        super().__init__()

        self.n_layers=1
        self.hidden_size=hidden_size
        
        self.embedding = nn.Embedding(output_size, embedding_size)        
        self.rnn = nn.GRU(embedding_size+hidden_size, hidden_size, 1, batch_first=True)        
        self.fc_out = nn.Linear(embedding_size + 2*hidden_size, output_size)

        self.init_weights()        
        
    def forward(self, target, hidden, context, teacher_forcing_ratio):
        """
        Parameters:
        -----------
            target: (B, T+1)
            hidden: (L*D, B, H)
            context: (L*D, B, H) 

        Returns:
        --------
            outputs: (B, K, T)
        """

        input = target[:,0] # (B)

        context = context.permute(1,0,2) # (B, L*D, H) 

        outputs = []
        for t in range(1, target.shape[1]):

            input = input.unsqueeze(1) # (B, 1)

            # embedded shape: (B, 1, E)       
            embedded = self.embedding(input)

            rnn_input = torch.cat((embedded, context), dim=2)

            # output shape: (B, 1, H), hidden shape: (L*D, B, H)
            output, hidden = self.rnn(rnn_input, hidden)

            # output shape: (B, E+2*H)
            output = torch.cat((embedded, hidden.permute(1,0,2), context), dim=2).squeeze(1)
        
            # output shape: (B, Out)
            output = self.fc_out(output)

            pred = output.argmax(1) # pred shape: (B)

            outputs.append(output)

            if random.random() < teacher_forcing_ratio:
                input = target[:,t] # use actual next token as next input
            else:
                input = pred

        outputs = torch.stack(outputs, dim=2) # (B, K, T)
        return outputs
    
    def init_weights(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)

    def init_hidden(self, batch_size):
        shape = (self.n_layers, batch_size, self.hidden_size)
        return torch.randn(shape).cuda()


class GRUDecoderWithAttention(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size):
        """Since this is an AutoEncoder Seq2Seq hybrid, the input size is the output size."""

        super().__init__()

        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = 1
   
        self.attention = Attention(output_size, hidden_size)           
        self.embedding = nn.Embedding(output_size, embedding_size)        
        self.rnn = nn.GRU(embedding_size+output_size, hidden_size, self.n_layers, batch_first=True)
        self.fc_out = nn.Linear(embedding_size+output_size+hidden_size, output_size)

        self.init_weights()
        
    def forward(self, target, hidden, teacher_forcing_ratio):
        """
        Parameters:
        -----------
            target: (B, T+1)
            hidden: previous decoder hidden (B, H_dec)

        Returns:
        --------
            outputs: (B, E, T)
            attentions: (B, T*(T+1)/2)
        """

        input = target[:,0] # (B)
        
        outputs, attentions = [torch.zeros((target.shape[0], 1, self.output_size)).cuda()], [] 
        for t in range(1, target.shape[1]):

            input = input.unsqueeze(1) # (B, 1)
       
            embedded = self.embedding(input) # (B, 1, E)

            # Look at the decoder output history
            output_history = torch.cat(outputs, dim=1) # (B, T, O)

            w, a = self.attention(hidden, output_history)
            # w: (B, 1, O), a: (B, T) 

            attentions.append(a)

            rnn_input = torch.cat((embedded, w), dim=2) # (B, 1, E+O)

            output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
            # output: (B, 1, H_dec), hidden: (1, B, H_dec)

            assert (output.permute(1,0,2) == hidden).all()

            hidden = hidden.squeeze(0) # (B, H_dec)

            output = self.fc_out(torch.cat((output, w, embedded), dim=2).squeeze(1)) # (B, E)

            pred = output.argmax(1) # pred shape: (B)

            outputs.append(output.unsqueeze(1)) # for (B, T, E)

            if random.random() < teacher_forcing_ratio:
                input = target[:,t] # use actual next token as next input
            else:
                input = pred

        outputs = torch.cat(outputs[1:], dim=1).permute(0, 2, 1) # (B, E, T) for loss
        attentions = torch.cat(attentions, dim=1) # (B, T*(T+1)/2)

        return outputs, attentions
    
    def init_weights(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)

    def init_hidden(self, batch_size):
        shape = (self.n_layers, batch_size, self.hidden_size)
        return torch.randn(shape).cuda()


class Attention(nn.Module):
    def __init__(self, output_dim, dec_O):
        super().__init__()
        
        self.attn = nn.Linear((dec_O+output_dim), dec_O)

        self.v = nn.Linear(dec_O, 1, bias=False)
        

    def forward(self, hidden, output_history):
        """
        hidden = (B, dec_H) prev dec hidden state
        output_history = (B, T, dec_O)

        w: (B, 1, dec_O)
        attention: (B, T)
        """
              
        src_len = output_history.shape[1]

        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1) # (B, T, dec_H) 

        attt = self.attn(torch.cat((hidden, output_history), dim = 2)) # (B, T, dec_H) 
        
        energy = torch.tanh(attt) # (B, T, dec_H)

        attention = self.v(energy).squeeze(2) # (B, T)

        attention = F.softmax(attention, dim=1).unsqueeze(1)  # (B,1,T)

        w =  torch.bmm(attention, output_history) # (B, 1, dec_O)
        
        return w, attention.squeeze(1)


class SimpleGRUDecoder(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size, n_layers):
        super().__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(output_size, embedding_size)        
        self.rnn = nn.GRU(embedding_size, hidden_size, num_layers=n_layers, batch_first=True)        
        self.fc_out = nn.Linear(hidden_size, output_size)

        self.init_weights()        
        
    def forward(self, target, hidden, teacher_forcing_ratio):
        """
        Parameters:
        -----------
            target: (B, T+1)
            hidden: (L, B, H) initial hidden state

        Returns:
        --------
            outputs: (B, K, T)
        """

        input = target[:,0] # (B)

        outputs = []
        for t in range(1, target.shape[1]):

            input = input.unsqueeze(1) # (B,1)

            # embedded shape: (B, 1, E)       
            embedded = self.embedding(input)

            # output shape: (B, 1, H)
            output, hidden = self.rnn(embedded, hidden)
        
            # output shape: (B, Out)
            output = self.fc_out(output.squeeze(1))

            outputs.append(output)

            pred = output.argmax(1)

            if random.random() < teacher_forcing_ratio:
                input = target[:,t] # use actual next token as next input
            else:
                input = pred

        outputs = torch.stack(outputs, dim=2) # (B, K, T)
        return outputs
    
    def init_weights(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)

    def init_hidden(self, batch_size):
        shape = (self.n_layers, batch_size, self.hidden_size)
        return torch.randn(shape).cuda()


class SimpleLSTMDecoder(nn.Module):
    def __init__(self, output_size, embedding_size, hidden_size, n_layers):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(output_size, embedding_size)        
        self.rnn = nn.LSTM(embedding_size, hidden_size, n_layers, batch_first=True)        
        self.fc_out = nn.Linear(hidden_size, output_size)

        self.init_weights()        
        
    def forward(self, target, hidden, cell, teacher_forcing_ratio):
        """
        Parameters:
        -----------
            target: (B, T+1)
            hidden, cell: (L, B, H) 

        Returns:
        --------
            outputs: (B, K, T)
        """

        input = target[:,0] # (B)

        outputs = []
        for t in range(1, target.shape[1]):

            input = input.unsqueeze(1) # (B,1)

            # embedded shape: (B, 1, E)       
            embedded = self.embedding(input)

            # output shape: (B, 1, H)
            output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        
            # output shape: (B, Out)
            output = self.fc_out(output.squeeze(1))

            outputs.append(output)

            pred = output.argmax(1)

            if random.random() < teacher_forcing_ratio:
                input = target[:,t] # use actual next token as next input
            else:
                input = pred

        outputs = torch.stack(outputs, dim=2) # (B, K, T)
        return outputs
    
    def init_weights(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param)

    def init_hidden_cell(self, batch_size):
        shape = (self.n_layers, batch_size, self.hidden_size)
        return torch.randn(shape).cuda(), torch.randn(shape).cuda()