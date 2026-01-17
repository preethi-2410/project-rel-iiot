import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        # output: (batch, seq_len, hidden_dim)
        # hidden: (num_layers, batch, hidden_dim)
        output, (hidden, cell) = self.lstm(x)
        
        # We use the last hidden state as the representation (embedding)
        # hidden[-1] is (batch, hidden_dim)
        return hidden[-1]

class Decoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x: (batch, seq_len, input_dim) - This input is usually repeated embedding
        output, (hidden, cell) = self.lstm(x)
        prediction = self.fc(output)
        return prediction

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 1, seq_len: int = 50):
        super(LSTMAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.encoder = Encoder(input_dim, hidden_dim, num_layers)
        self.decoder = Decoder(hidden_dim, hidden_dim, input_dim, num_layers)
        
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        
        # Encode
        # embedding: (batch, hidden_dim)
        embedding = self.encoder(x)
        
        # Prepare input for decoder
        # We repeat the embedding seq_len times to reconstruct the sequence
        # (batch, seq_len, hidden_dim)
        decoder_input = embedding.unsqueeze(1).repeat(1, self.seq_len, 1)
        
        # Decode
        reconstruction = self.decoder(decoder_input)
        
        return reconstruction
