import numpy as np
import polars as pl
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import gc

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cuda'
print(device)

num_files = 9
dataset = pl.concat([pl.read_parquet(f"./train_data/train_{i+3}.parquet") for i in range(num_files)])

# Creating lag features 

lags = dataset.select(pl.col(["date_id", "symbol_id"] + [f"target_{idx}" for idx in range(1, 10)]))
lags = lags.rename({ f"target_{idx}" : f"lag_target_{idx}" for idx in range(1, 10)})
lags = lags.with_columns(
    date_id = pl.col('date_id') + 1,  # lagged by 1 day
    )
lags = lags.group_by(["date_id", "symbol_id"], maintain_order=True).last()
dataset = dataset.join(lags, on=["date_id", "symbol_id"],  how="left")

dataset = dataset.fill_null(0)

categorical_features = ["feature_09", "feature_10", "feature_11"]
dataset = dataset.to_dummies(columns=categorical_features)

# df_head = dataset.slice(0, 20000)

# # Save to CSV
# df_head.write_csv("output.csv")

feature_columns = [col for col in dataset.columns if not col.startswith("target_")]

# Split dataset into train and test
x_train = dataset.select(feature_columns)
y_train = dataset.select(["target_1", "target_2"])


# Print shapes
print(x_train.shape)
print(y_train.shape)

gc.collect()

print(feature_columns)

del dataset
gc.collect()

def to_tensor(df, dtype=torch.float32):
    return torch.tensor(df.to_numpy(), dtype=dtype)  # Use .to_numpy() instead of .values

# Convert Polars DataFrame to PyTorch tensors
x_train = to_tensor(x_train).unsqueeze(1)  # (batch_size, seq_length=1, input_size)
y_train = to_tensor(y_train)  # (batch_size, 1)

gc.collect()

# Print shapes
print(x_train.shape)
print(y_train.shape)

# print(x_train)
# print(y_train)
# print(x_test)
# print(y_test)

# Create PyTorch DataLoader
batch_size = 10000

train_dataset = torch.utils.data.TensorDataset(x_train, y_train)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)


# loss function

class R2Loss(nn.Module):
    def __init__(self):
        super(R2Loss, self).__init__()

    def forward(self, y_pred, y_true, weights):        
        ss_total = torch.sum(weights * (y_true ** 2), dim=0)
        ss_residual = torch.sum(weights * ((y_true - y_pred) ** 2), dim=0)
        r2 = 1 - (ss_residual / (ss_total + 1e-8))
        return 1 - torch.mean(r2)
    
class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=2, dropout=0.2):
        super(StockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return torch.clamp(out, -5, 5)
    
# Initialize model
input_size = x_train.shape[2]
model = StockLSTM(input_size)

# Move model to device (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
max_grad_norm = 5.0
print(model)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)  # Better weight scaling
        if m.bias is not None:
            nn.init.zeros_(m.bias)

model.apply(init_weights)  # Apply to the entire model

lr = 0.001
epochs = 60
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = R2Loss() 

print("Training started...")

train_losses = []

for epoch in range(1, epochs + 1):
    print("epoch:- ", epoch)
    model.train()
    total_loss = 0

    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        weights = x_batch[:, :, 3]
        weights = weights.to(device)

        # Forward pass
        outputs = model(x_batch)

        loss = criterion(outputs, y_batch, weights)
        # print(loss.item()/batch_size)
        # Backward pass

        optimizer.zero_grad()
        loss.backward()

        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:  # Some parameters may not have gradients
                param_norm = param.grad.norm(2)  # L2 norm of gradients
                total_norm += param_norm.item() ** 2  # Sum squared norms
    
        total_norm = total_norm ** 0.5  # Take the square root
        # print(f"Gradient Norm: {total_norm:.6f}")
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    
    train_losses.append(avg_loss)
    print(f"Epoch [{epoch}/{epochs}], Loss: {avg_loss}")

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'criterion_state_dict': criterion.state_dict(),
        'full_feature_set': feature_columns,  # Save feature names
    }, f"LSTM_{epoch}.pth")

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'criterion_state_dict': criterion.state_dict(),
        'full_feature_set': feature_columns,  # Save feature names
    }, f"LSTM.pth")


