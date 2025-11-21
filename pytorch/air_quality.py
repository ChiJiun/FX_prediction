from ucimlrepo import fetch_ucirepo
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# fetch dataset
air_quality = fetch_ucirepo(id=360)

# data (as pandas dataframes)
X = air_quality.data.features
y = air_quality.data.targets

# metadata
# print(air_quality.metadata)

# variable information
# print(air_quality.variables)

# 1. 數據加載及前處理
def preprocess_data(X, y):
    # 合併特徵與目標
    data = pd.concat([X, y], axis=1)

    # 建立時間索引：合併 Date 和 Time
    data['DateTime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%d/%m/%Y %H.%M.%S', errors='coerce')
    nan_count = data['DateTime'].isna().sum()
    nan_ratio = nan_count / len(data)
    print(f"DateTime NaN count: {nan_count}, ratio: {nan_ratio:.2%}")
    # 如果DateTime有NaN太多，用linear插值；否則用time插值
    if nan_ratio > 0.1:  # 如果NaN > 10%，用linear
        print("Too many NaN in DateTime, using linear interpolation")
        interp_method = 'linear'
        data = data.drop(columns=['DateTime'])  # 不設索引
    else:
        data = data.dropna(subset=['DateTime'])
        data = data.set_index('DateTime').sort_index()
        interp_method = 'time'

    # 選擇數值特徵（排除 Date 和 Time）
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data_numeric = data[numeric_cols]

    # 用插值處理缺失值
    data_numeric = data_numeric.interpolate(method=interp_method)

    # 用 Z-Score 找出異常點（絕對值 > 3）
    z_scores = np.abs((data_numeric - data_numeric.mean()) / data_numeric.std())
    outliers = z_scores > 3
    data_numeric[outliers] = np.nan

    # 再用插值處理異常點
    data_numeric = data_numeric.interpolate(method=interp_method)

    # 正規化
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_numeric)

    return data_scaled, scaler

# 2. 定義資料集和 DataLoader，包含滑動窗口生成特徵與標籤
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length=24, target_col=-1):
        self.data = data
        self.seq_length = seq_length
        self.target_col = target_col

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length, :-1]  # 特徵，形狀: (seq_length, num_features)
        y = self.data[idx + self.seq_length, self.target_col]  # 目標（最後一欄），形狀: scalar
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32) 

class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i]
            losses.append(torch.max((q-1) * errors, q * errors).unsqueeze(1))
        loss = torch.cat(losses, dim=1).mean()
        return loss

# 3. 定義模型（LSTM）
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=3, output_size=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def main():
    # 數據前處理
    data_scaled, scaler = preprocess_data(X, y)

    # 建立資料集
    seq_length = 24  # window size
    forecast_horizon = 1  # 預測下一個時間點
    step_size = 1  # 步長
    dataset = TimeSeriesDataset(data_scaled, seq_length=seq_length)

    # 數據分割：70% 訓練，15% 驗證，15% 測試
    total_len = len(dataset)
    train_end = int(0.7 * total_len)
    val_end = int(0.85 * total_len)  # 70% + 15%

    train_dataset = torch.utils.data.Subset(dataset, range(0, train_end))
    val_dataset = torch.utils.data.Subset(dataset, range(train_end, val_end))
    test_dataset = torch.utils.data.Subset(dataset, range(val_end, total_len))

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 4. 建立模型、損失函數和優化器
    input_size = data_scaled.shape[1] - 1  # 特徵數量
    model = LSTMModel(input_size=input_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 5. 訓練參數
    num_epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 6. 迴圈開始訓練
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 驗證
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                output = model(batch_x)
                loss = criterion(output.squeeze(), batch_y)
                val_loss += loss.item()

        val_loss_avg = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss_avg:.4f}")

        # 保存最佳模型
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Best model saved at epoch {epoch+1}")

    # 測試最佳模型
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            output = model(batch_x)
            loss = criterion(output.squeeze(), batch_y)
            test_loss += loss.item()
            all_preds.extend(output.squeeze().cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())

    test_loss_avg = test_loss / len(test_loader)
    mae = mean_absolute_error(all_targets, all_preds)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    r2 = r2_score(all_targets, all_preds)

    print(f"Test Loss (MSE): {test_loss_avg:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test R²: {r2:.4f}")

    # 視覺化：實測 vs 預測 曲線
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(all_targets[:200], label='True', alpha=0.7)  # 只畫前200點
    plt.plot(all_preds[:200], label='Predicted', alpha=0.7)
    plt.title('True vs Predicted')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()

    # 誤差時間趨勢圖
    errors = np.array(all_targets) - np.array(all_preds)
    plt.subplot(1, 2, 2)
    plt.plot(errors[:200], label='Error', color='red')
    plt.axhline(y=0, color='black', linestyle='--')
    plt.title('Prediction Errors Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Error')
    plt.legend()

    plt.tight_layout()
    plt.savefig('evaluation_plots.png')
    plt.show()

def backtesting(data, seq_length=24, train_ratio=0.7, step=1):
    """
    回測：滾動預測，檢視模型穩定性
    """
    n = len(data)
    train_size = int(n * train_ratio)
    predictions = []
    actuals = []

    for i in range(train_size, n - seq_length, step):
        # 訓練數據：0到i
        train_data = data[:i]
        if len(train_data) < seq_length + 10:  # 至少有點數據
            continue

        # 簡單訓練（這裡簡化，用線性模型代替LSTM以節省時間）
        from sklearn.linear_model import LinearRegression
        X_train = []
        y_train = []
        for j in range(len(train_data) - seq_length):
            X_train.append(train_data[j:j+seq_length, :-1].flatten())
            y_train.append(train_data[j+seq_length, -1])
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        model = LinearRegression()
        model.fit(X_train, y_train)

        # 預測下一個
        x_pred = data[i:i+seq_length, :-1].flatten().reshape(1, -1)
        pred = model.predict(x_pred)[0]
        actual = data[i+seq_length, -1]

        predictions.append(pred)
        actuals.append(actual)

    # 計算回測指標
    mae_bt = mean_absolute_error(actuals, predictions)
    rmse_bt = np.sqrt(mean_squared_error(actuals, predictions))
    r2_bt = r2_score(actuals, predictions)

    print("\nBacktesting Results:")
    print(f"MAE: {mae_bt:.4f}")
    print(f"RMSE: {rmse_bt:.4f}")
    print(f"R²: {r2_bt:.4f}")

    # 畫回測圖
    plt.figure(figsize=(10, 5))
    plt.plot(actuals[:100], label='Actual', alpha=0.7)
    plt.plot(predictions[:100], label='Predicted', alpha=0.7)
    plt.title('Backtesting: Rolling Predictions')
    plt.xlabel('Prediction Step')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig('backtesting_plot.png')
    plt.show()

    return predictions, actuals

if __name__ == "__main__":
    main()
    # 回測
    data_scaled, _ = preprocess_data(X, y)
    backtesting(data_scaled)