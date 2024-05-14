import torch
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
#from src.data.data_preprocessing import updated_df
import pandas as pd


df = pd.read_excel('/Users/seanwhite/OneDrive - University of Greenwich/Documents/Group Project/group_project_code/src/data/Ocado Stock & Trends.xlsx')

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[['Close', 'Open', 'Low', 'High', 'Flow', 'Covid', 'Online Shop', 'Next Day Close']])

# Convert to PyTorch tensors
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length, -1]
        xs.append(x)
        ys.append(y)
    return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32)

seq_length = 50  # Using 10 days of data to predict the next day
X, y = create_sequences(scaled_data, seq_length)

# Split into training and testing sets
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Define the LSTM model with increased complexity
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=2, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        predictions = self.linear(lstm_out[:, -1])
        return predictions

input_size = X.shape[2]
hidden_layer_size = 100  # Increased hidden layer size
output_size = 1

model = LSTMModel(input_size, hidden_layer_size, output_size)

loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = loss_function(y_pred, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch} Loss: {loss.item()}')

model.eval()
with torch.no_grad():
    y_test_pred = model(X_test)

# Inverse transform the predictions
y_test_pred = scaler.inverse_transform(np.concatenate((np.zeros((y_test_pred.shape[0], 7)), y_test_pred.numpy()), axis=1))[:, -1]
y_test = scaler.inverse_transform(np.concatenate((np.zeros((y_test.shape[0], 7)), y_test.numpy().reshape(-1, 1)), axis=1))[:, -1]

plt.plot(y_test, label='True')
plt.plot(y_test_pred, label='Predicted')
plt.legend()
plt.show()
