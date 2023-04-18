import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import torch
import torch.nn as nn
import torch.optim as optim

# Download the data
data = yf.download(tickers='^RUI', start='2012-03-11', end='2022-07-10')

# Add indicators
data['RSI'] = ta.rsi(data.Close, length=15)
data['EMAF'] = ta.ema(data.Close, length=20)
data['EMAM'] = ta.ema(data.Close, length=100)
data['EMAS'] = ta.ema(data.Close, length=150)

data['Target'] = data['Adj Close'] - data.Open
data['Target'] = data['Target'].shift(-1)

data['TargetClass'] = [1 if data.Target[i] > 0 else 0 for i in range(len(data))]

data['TargetNextClose'] = data['Adj Close'].shift(-1)

# Drop unused columns
data.dropna(inplace=True)
data = data[['High', 'Low', 'Open', 'RSI', 'EMAF', 'EMAM', 'EMAS', 'TargetNextClose']]

# Scale the data
sc = MinMaxScaler(feature_range=(0,1))
data_set_scaled = sc.fit_transform(data)

# Create input features and targets
X, y = [], []
backcandles = 30
for j in range(data_set_scaled.shape[1] - 1):
    X.append([])
    for i in range(backcandles, data_set_scaled.shape[0]):
        X[j].append(data_set_scaled[i-backcandles:i, j])
    y.append(data_set_scaled[backcandles:, -1])
X, y = np.array(X), np.array(y).T

# Split the data into train and test sets
splitlimit = int(len(X)*0.8)
X_train, X_test = X[:splitlimit], X[splitlimit:]
y_train, y_test = y[:splitlimit], y[splitlimit:]

# Convert data to PyTorch tensors
X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()

# Define the model architecture
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

# Initialize the model, loss function, and optimizer
model = LSTMModel(input_size=X_train.shape[2], hidden_size=150, output_size=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

# Train the model
epochs = 30
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs.squeeze(), y_train)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, loss.item()))

# Evaluate the model on the test set
with torch.no_grad():
    y_pred = model(X_test).squeeze().numpy

