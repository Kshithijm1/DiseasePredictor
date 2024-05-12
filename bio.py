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

# AttentionLayer: Applies attention mechanism over LSTM outputs for better feature extraction.
class AttentionLayer(nn.Module):
    def __init__(self):
        super(AttentionLayer, self).__init__()
        self.softmax = nn.Softmax(dim=1)  # Softmax layer to create attention weights.

    def forward(self, lstm_output):
        attention_weights = self.softmax(lstm_output)  # Generate attention weights.
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)  # Create context vector as weighted sum.
        return context_vector, attention_weights

# LSTMWithAttention: Combines LSTM layers with an attention mechanism for sequence classification.
class LSTMWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(LSTMWithAttention, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=5, embedding_dim=input_dim)  # Embedding layer for sequence input.
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)  # LSTM layer with bidirectional setting.
        self.attention = AttentionLayer()  # Attention layer integration.
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)  # Fully connected layer for dimension reduction.
        self.relu = nn.ReLU()  # ReLU activation function.
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # Final output layer.

    def forward(self, x):
        embedded_sequences = self.embedding(x)  # Embedding the input sequences.
        lstm_output, _ = self.lstm(embedded_sequences)  # Getting LSTM outputs.
        context_vector, attention_weights = self.attention(lstm_output)  # Applying attention mechanism.
        x = self.relu(self.fc1(context_vector))  # Applying ReLU activation.
        x = torch.sigmoid(self.fc2(x)).squeeze()  # Output with sigmoid activation for binary classification.
        return x

# Load sequences and labels from a FASTA file, parsing additional metadata.
def load_sequences_and_labels(file_path):
    sequences = []
    labels = []
    metadata = []
    for record in SeqIO.parse(file_path, 'fasta'):  # Parse each record from the FASTA file.
        parts = record.description.split('|')
        label = int(parts[1].strip().split()[1])
        age = int(parts[2].strip().split()[1])
        sex = parts[3].strip().split()[1]

        sequences.append(str(record.seq).upper())  # Convert sequence to uppercase string.
        labels.append(label)
        metadata.append({'age': age, 'sex': sex})

    return sequences, labels, metadata

# Encode DNA sequences into numerical format suitable for model input.
def encode_sequences(sequences, max_length=None):
    mapping = {'A': 1, 'C': 2, 'G': 3, 'T': 4, 'N': 0}  # Mapping for nucleotides to integers.
    if not max_length:
        max_length = max(len(seq) for seq in sequences)  # Automatically determine the maximum sequence length.

    encoded_sequences = []
    for seq in sequences:
        encoded_seq = [mapping.get(base, 0) for base in seq]  # Encode each base of the sequence.
        padded_seq = encoded_seq + [0] * (max_length - len(encoded_seq))  # Pad sequences to the maximum length.
        encoded_sequences.append(padded_seq)

    return np.array(encoded_sequences, dtype=np.int64)

# Function to train the LSTM model with attention, including validation.
def train_model(model, sequences, labels, optimizer, criterion, epochs=10, validation_split=0.2):
    sequences = torch.tensor(sequences, dtype=torch.long)  # Convert sequences to torch tensor.
    labels = torch.tensor(labels, dtype=torch.float32).view(-1)  # Convert labels to torch tensor and flatten.

    # Split data into training and validation sets.
    train_seq, valid_seq, train_lbl, valid_lbl = train_test_split(sequences, labels, test_size=validation_split, shuffle=True)
    train_dataset = TensorDataset(train_seq, train_lbl)  # Create dataset for training.
    valid_dataset = TensorDataset(valid_seq, valid_lbl)  # Create dataset for validation.

    for epoch in range(epochs):  # Training loop.
        model.train()  # Set model to training mode.
        total_train_loss = 0
        for seq, lbl in DataLoader(train_dataset, batch_size=10, shuffle=True):
            optimizer.zero_grad()  # Clear gradients.
            output = model(seq)  # Forward pass.
            loss = criterion(output, lbl)  # Compute loss.
            loss.backward()  # Backpropagation.
            optimizer.step()  # Update weights.
            total_train_loss += loss.item()  # Sum up loss for diagnostics.

        average_train_loss = total_train_loss / len(train_dataset)  # Calculate average loss.

        model.eval()  # Set model to evaluation mode for validation.
        total_val_loss = 0
        with torch.no_grad():  # Disable gradient computation for evaluation.
            for seq, lbl in DataLoader(valid_dataset, batch_size=10):
                output = model(seq)
                val_loss = criterion(output, lbl)
                total_val_loss += val_loss.item()

        average_val_loss = total_val_loss / len(valid_dataset)
        print(f'Epoch {epoch+1}, Training Loss: {average_train_loss}, Validation Loss: {average_val_loss}')

# Setup a Dash application to visualize sequence data and predictions interactively.
def setup_dash_app(sequences, metadata, model):
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])  # Initialize Dash app with Bootstrap styling.
    app.layout = html.Div([
        dcc.Graph(id='sequence-graph'),  # Graph component to display sequence profiles.
        dcc.Slider(  # Slider component to select different sequences.
            id='sequence-slider',
            min=0,
            max=len(sequences) - 1,
            value=0,
            marks={i: 'Seq {}'.format(i+1) for i in range(len(sequences))},
            step=None
        ),
        html.Div(id='sequence-output')  # Div to display selected sequence information.
    ])

    @app.callback(  # Setup callback to update graph and sequence information based on slider value.
        Output('sequence-graph', 'figure'),
        Output('sequence-output', 'children'),
        Input('sequence-slider', 'value')
    )
    def update_output(slider_value):
        seq = sequences[slider_value]  # Get the selected sequence.
        meta = metadata[slider_value]  # Get metadata for the selected sequence.
        input_tensor = torch.tensor(encode_sequences([seq]), dtype=torch.int64)  # Encode sequence for prediction.
        prediction = model(input_tensor).item()  # Predict disease risk.
        fig = px.line(x=list(range(len(seq))), y=[ord(c) for c in seq], title=f'Base Profile for Sequence {slider_value+1}')  # Create a plotly line graph of the sequence.
        return fig, f'Selected Sequence: {seq}, Age: {meta["age"]}, Sex: {meta["sex"]}, Disease Risk: {prediction:.2f}'

    return app

# Main execution block to run the Dash application.
if __name__ == "__main__":
    file_path = 'example.fasta'
    if os.path.exists(file_path):
        sequences, labels, metadata = load_sequences_and_labels(file_path)  # Load data from file.
        max_sequence_length = max(len(seq) for seq in sequences)  # Determine max length for padding.
        encoded_sequences = encode_sequences(sequences, max_sequence_length)  # Encode all sequences.
        model = LSTMWithAttention(input_dim=64, hidden_dim=128, output_dim=1, num_layers=3)  # Initialize model.
        optimizer = optim.Adam(model.parameters())  # Setup optimizer.
        criterion = nn.BCELoss()  # Setup loss function.
        train_model(model, encoded_sequences, labels, optimizer, criterion, epochs=200)  # Train model.

        app = setup_dash_app(sequences, metadata, model)  # Setup Dash application.
        app.run_server(debug=True, use_reloader=False)  # Run server with Dash app.
    else:
        print(f"Error: File {file_path} does not exist.")
