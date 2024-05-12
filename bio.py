import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from Bio import SeqIO
import os
from sklearn.model_selection import train_test_split
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import dash_bootstrap_components as dbc

# Define the Attention Layer
class AttentionLayer(nn.Module):
    def __init__(self):
        super(AttentionLayer, self).__init__()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, lstm_output):
        attention_weights = self.softmax(lstm_output)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector, attention_weights

# Define the LSTM with Attention model
class LSTMWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(LSTMWithAttention, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=5, embedding_dim=input_dim)
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.attention = AttentionLayer()
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded_sequences = self.embedding(x)
        lstm_output, _ = self.lstm(embedded_sequences)
        context_vector, attention_weights = self.attention(lstm_output)
        x = self.relu(self.fc1(context_vector))
        x = torch.sigmoid(self.fc2(x)).squeeze()  # Added sigmoid activation here
        return x

# Load sequences and labels for training
def load_sequences_and_labels(file_path):
    sequences = []
    labels = []
    metadata = []
    for record in SeqIO.parse(file_path, 'fasta'):
        parts = record.description.split('|')
        label = int(parts[1].strip().split()[1])
        age = int(parts[2].strip().split()[1])
        sex = parts[3].strip().split()[1]

        sequences.append(str(record.seq).upper())
        labels.append(label)
        metadata.append({'age': age, 'sex': sex})

    return sequences, labels, metadata

# Encode sequences for model input
def encode_sequences(sequences, max_length=None):
    mapping = {'A': 1, 'C': 2, 'G': 3, 'T': 4, 'N': 0}
    if not max_length:
        max_length = max(len(seq) for seq in sequences)  # Find the longest sequence

    encoded_sequences = []
    for seq in sequences:
        encoded_seq = [mapping.get(base, 0) for base in seq]
        padded_seq = encoded_seq + [0] * (max_length - len(encoded_seq))
        encoded_sequences.append(padded_seq)

    return np.array(encoded_sequences, dtype=np.int64)

# Training the model with validation and adjusted loss function
def train_model(model, sequences, labels, optimizer, criterion, epochs=10, validation_split=0.2):
    sequences = torch.tensor(sequences, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.float32).view(-1)  # Make sure labels are flat

    train_seq, valid_seq, train_lbl, valid_lbl = train_test_split(sequences, labels, test_size=validation_split, shuffle=True)
    train_dataset = TensorDataset(train_seq, train_lbl)
    valid_dataset = TensorDataset(valid_seq, valid_lbl)

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for seq, lbl in DataLoader(train_dataset, batch_size=10, shuffle=True):
            optimizer.zero_grad()
            output = model(seq)
            loss = criterion(output, lbl)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        average_train_loss = total_train_loss / len(train_dataset)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for seq, lbl in DataLoader(valid_dataset, batch_size=10):
                output = model(seq)
                val_loss = criterion(output, lbl)
                total_val_loss += val_loss.item()

        average_val_loss = total_val_loss / len(valid_dataset)
        print(f'Epoch {epoch+1}, Training Loss: {average_train_loss}, Validation Loss: {average_val_loss}')

# Setup Dash application for visualization
def setup_dash_app(sequences, metadata, model):
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.layout = html.Div([
        dcc.Graph(id='sequence-graph'),
        dcc.Slider(
            id='sequence-slider',
            min=0,
            max=len(sequences) - 1,
            value=0,
            marks={i: 'Seq {}'.format(i+1) for i in range(len(sequences))},
            step=None
        ),
        html.Div(id='sequence-output')
    ])

    @app.callback(
        Output('sequence-graph', 'figure'),
        Output('sequence-output', 'children'),
        Input('sequence-slider', 'value')
    )
    def update_output(slider_value):
        seq = sequences[slider_value]
        meta = metadata[slider_value]
        input_tensor = torch.tensor(encode_sequences([seq]), dtype=torch.int64)
        prediction = model(input_tensor).item()
        fig = px.line(x=list(range(len(seq))), y=[ord(c) for c in seq], title=f'Base Profile for Sequence {slider_value+1}')
        return fig, f'Selected Sequence: {seq}, Age: {meta["age"]}, Sex: {meta["sex"]}, Disease Risk: {prediction:.2f}'

    return app

# Main execution block
if __name__ == "__main__":
    file_path = 'example.fasta'
    if os.path.exists(file_path):
        sequences, labels, metadata = load_sequences_and_labels(file_path)
        max_sequence_length = max(len(seq) for seq in sequences)
        encoded_sequences = encode_sequences(sequences, max_sequence_length)
        model = LSTMWithAttention(input_dim=64, hidden_dim=128, output_dim=1, num_layers=3)
        optimizer = optim.Adam(model.parameters())
        criterion = nn.BCELoss()  # Changed to BCELoss
        train_model(model, encoded_sequences, labels, optimizer, criterion, epochs=200)

        app = setup_dash_app(sequences, metadata, model)
        app.run_server(debug=True, use_reloader=False)
    else:
        print(f"Error: File {file_path} does not exist.")
