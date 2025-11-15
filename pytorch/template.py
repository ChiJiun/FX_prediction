import torch
from torch.utils.data import DataLoader, TensorDataset

# 1. 定義模型（線性回歸範例）
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


def main():
    # 2. 建立模型、損失函數和優化器
    model = Net()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # 建立示範資料與 DataLoader（範例可替換成實際資料）
    num_samples = 128
    X = torch.randn(num_samples, 10)
    y = torch.randn(num_samples, 1)
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 訓練參數
    num_epochs = 5

    # 3. 迴圈開始訓練
    for epoch in range(num_epochs):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()        # 清空梯度
            output = model(batch_x)      # forward
            loss = criterion(output, batch_y)  # 計算損失
            loss.backward()              # 反向傳播
            optimizer.step()             # 優化器更新權重

        # 每個 epoch 顯示最後一個 batch 的 loss 作為簡單監控
        print(f"Epoch {epoch+1}/{num_epochs}, loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()
    print("CUDA available:", torch.cuda.is_available())     # 檢查能否使用CUDA