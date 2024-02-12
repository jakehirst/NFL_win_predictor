import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

'''if both of thest statements are true, then pytorch is correctly installed with GPU available!'''
print(torch.backends.mps.is_available())
print(torch.backends.mps.is_built())

# Define the LSTM model
class GameOutcomeLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1, num_layers=2):
        super(GameOutcomeLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # Sigmoid activation
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        
        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        
        # Index hidden state of last time step
        out = self.fc(out[:, -1, :]) 
        out = self.sigmoid(out)
        return out

def run_epoch(loader, model, is_training=True):
    """Runs a training or evaluation epoch and returns loss and accuracy."""
    if is_training:
        model.train()
    else:
        model.eval()
    
    total_loss = 0
    total_accuracy = 0
    total_samples = 0
    
    with torch.set_grad_enabled(is_training):
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze()
            
            loss = criterion(outputs, labels.float())
            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item() * inputs.size(0)
            preds = outputs.round()
            total_accuracy += (preds == labels).sum().item()
            total_samples += inputs.size(0)
    
    avg_loss = total_loss / total_samples
    avg_accuracy = total_accuracy / total_samples
    
    return avg_loss, avg_accuracy



'''***************  now trying with real data  ***************'''
# Load the arrays
features = np.load('/Users/jakehirst/Desktop/sportsbetting/nfl/data/features.npy')
labels = np.load('/Users/jakehirst/Desktop/sportsbetting/nfl/data/labels.npy')
game_log = np.load('/Users/jakehirst/Desktop/sportsbetting/nfl/data/game_log.npy')

# Split the data so that the last 1000 games are in the test set
test_size = 1000  # Define the size of the test set (just the last 1000 games of the dataset)

# Ensure the division for train and test maintains the order, especially important for time series data
features_train, features_test = features[:-test_size], features[-test_size:]
labels_train, labels_test = labels[:-test_size], labels[-test_size:]
game_log_train, game_log_test = game_log[:-test_size], game_log[-test_size:]

features_train.shape, features_test.shape, labels_train.shape, labels_test.shape, game_log_train.shape, game_log_test.shape

# Convert the arrays to PyTorch tensors
features_train_tensor = torch.tensor(features_train, dtype=torch.float)
features_test_tensor = torch.tensor(features_test, dtype=torch.float)
labels_train_tensor = torch.tensor(labels_train, dtype=torch.long)
labels_test_tensor = torch.tensor(labels_test, dtype=torch.long)

# Create TensorDatasets
train_dataset = TensorDataset(features_train_tensor, labels_train_tensor)
test_dataset = TensorDataset(features_test_tensor, labels_test_tensor)

# Create DataLoaders
batch_size = 500  # Adjust as needed
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Return DataLoader sizes as a confirmation
len(train_loader), len(test_loader)

# Parameters
input_dim = features_train_tensor.shape[-1]  # Number of features per game
hidden_dim = 100  # Number of hidden units #COMMENT dont know what this is yet
num_layers = 2  # Number of LSTM layers
output_dim = 1  # Output dimension

# Instantiate the model
model = GameOutcomeLSTM(input_dim, hidden_dim, output_dim, num_layers)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Move model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Number of epochs
num_epochs = 100  # Adjust as needed

# Main training loop
for epoch in range(num_epochs):
    train_loss, train_accuracy = run_epoch(train_loader, model, is_training=True)
    test_loss, test_accuracy = run_epoch(test_loader, model, is_training=False)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
          f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')





'''example of how to run with dummy data'''
# # Parameters
# input_dim = 8  # Number of features per game
# hidden_dim = 100  # Number of hidden units
# num_layers = 2  # Number of LSTM layers
# output_dim = 1  # Output dimension

# # Instantiate the model
# model = GameOutcomeLSTM(input_dim, hidden_dim, output_dim, num_layers)

# # Loss and optimizer
# criterion = nn.BCELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)


# # Dummy input data: batch_size x seq_length x feature_size
# # For example, 64 games, each sequence has 10 games, each game has 8 features
# batch_size = 64 #number of predicted games to learn on per batch
# seq_length = 10 #number of games that will be fed into the lstm
# dummy_input = torch.randn(batch_size, seq_length, input_dim)

# # Dummy target data: batch_size
# dummy_target = torch.randint(0, 2, (batch_size, output_dim)).float()

# # Example of training step
# model.train()  # Set the model to training mode
# optimizer.zero_grad()  # Clear existing gradients
# output = model(dummy_input)  # Forward pass
# loss = criterion(output, dummy_target)  # Compute loss
# loss.backward()  # Backpropagation
# optimizer.step()  # Update weights