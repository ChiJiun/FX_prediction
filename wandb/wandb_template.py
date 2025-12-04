import random
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import wandb

# Start a new wandb run to track this script.
run = wandb.init(
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": 0.01,
        "architecture": "3LayerNN",
        "dataset": "Diabetes",
        "epochs": 100,
    },
)

# Load and prepare data
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Define model
class Simple3LayerNN(nn.Module):
    def __init__(self, input_dim):
        super(Simple3LayerNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = Simple3LayerNN(input_dim=X_train.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Real training loop
train_losses = []
for epoch in range(100):
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())
    
    # Evaluate on test set for metrics
    model.eval()
    with torch.no_grad():
        test_output = model(X_test)
        test_loss = criterion(test_output, y_test)
        mse = mean_squared_error(y_test.numpy(), test_output.numpy())
        r2 = r2_score(y_test.numpy(), test_output.numpy())
    model.train()
    
    # Log real metrics to wandb
    run.log({"epoch": epoch, "train_loss": loss.item(), "test_loss": test_loss.item(), "mse": mse, "r2": r2})
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}, MSE: {mse:.4f}, R²: {r2:.4f}")

# Finish the run and upload any remaining data.
run.finish()