from sklearn.datasets import load_diabetes                  # 匯入sklearn的糖尿病數據集載入函數
from sklearn.model_selection import train_test_split        # 匯入訓練測試集分割函數
import torch                                                # 匯入PyTorch庫
import torch.nn as nn                                       # 匯入神經網路模組
import torch.optim as optim                                 # 匯入優化器模組
from sklearn.metrics import mean_squared_error, r2_score    # 匯入評估指標函數
import matplotlib.pyplot as plt                             # 匯入matplotlib用於視覺化

diabetes = load_diabetes()  # 載入糖尿病數據集：獲取包含特徵和目標的數據
X = diabetes.data           # 獲取特徵數據：X是(442, 10)的特徵矩陣
y = diabetes.target         # 獲取目標變數：y是(442,)的目標向量
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 分割訓練和測試集，測試集佔20%：確保重現性，測試模型泛化

X_train = torch.tensor(X_train, dtype=torch.float32)                # 將訓練特徵轉換為PyTorch張量：PyTorch需要張量格式進行計算
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)    # 將訓練目標轉換為張量並調整形狀：從(353,)變為(353, 1)，匹配輸出
X_test = torch.tensor(X_test, dtype=torch.float32)                  # 將測試特徵轉換為PyTorch張量：同上
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)      # 將測試目標轉換為張量並調整形狀：同上

class Simple3LayerNN(nn.Module):  # 定義一個簡單的三層神經網路類：繼承nn.Module，建立自定義模型
    def __init__(self, input_dim):                          # 初始化函數，接收輸入維度 (input_dim: int)：設定層參數
        super(Simple3LayerNN, self).__init__()              # 呼叫父類初始化：確保正確繼承
        self.fc1 = nn.Linear(input_dim, 32)                 # 第一層全連接層：輸入 (batch, input_dim) -> 輸出 (batch, 32)：學習特徵組合，維度從 input_dim 擴展到 32
        self.fc2 = nn.Linear(32, 16)                        # 第二層全連接層：輸入 (batch, 32) -> 輸出 (batch, 16)：進一步抽象，維度從 32 縮減到 16
        self.fc3 = nn.Linear(16, 1)                         # 第三層全連接層：輸入 (batch, 16) -> 輸出 (batch, 1)：輸出預測值，維度從 16 縮減到 1
        self.relu = nn.ReLU()                               # ReLU激活函數：引入非線性，避免梯度消失
    def forward(self, x):           # 前向傳播函數，輸入 x: (batch, input_dim)：定義數據流
        x = self.relu(self.fc1(x))  # 第一層後應用ReLU，輸出: (batch, 32)：激活後傳遞，形狀轉換通過權重矩陣
        x = self.relu(self.fc2(x))  # 第二層後應用ReLU，輸出: (batch, 16)：同上，形狀從 32 變 16
        x = self.fc3(x)             # 第三層輸出: (batch, 1)：最終預測，形狀從 16 變 1
        return x                    # 返回輸出: (batch, 1)：模型結果

model = Simple3LayerNN(input_dim=X_train.shape[1])      # 實例化模型，輸入維度為訓練特徵數：建立模型實例
criterion = nn.MSELoss()                                # 定義均方誤差損失函數：適合回歸任務，衡量預測誤差
optimizer = optim.Adam(model.parameters(), lr=0.01)     # 定義Adam優化器，學習率0.01：高效更新參數

train_losses = []                                           # 用於存儲每個epoch的訓練損失：便於後續視覺化
for epoch in range(100):                                    # 訓練100個epoch：epoch是訓練的一次完整遍歷，即模型處理整個訓練數據集一次
    optimizer.zero_grad()                                   # 清空梯度：避免累積梯度
    output = model(X_train)                                 # 模型前向傳播：計算預測
    loss = criterion(output, y_train)                       # 計算損失：比較預測與真實
    loss.backward()                                         # 反向傳播：計算梯度
    optimizer.step()                                        # 更新參數：根據梯度調整權重
    train_losses.append(loss.item())                        # 收集損失：添加到列表
    if epoch % 10 == 0:                                     # 每10個epoch印出一次損失：監控進展
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")    # 印出當前epoch和損失：顯示訓練狀態

# 測試評估
model.eval()                                                        # 設定模型為評估模式：禁用dropout等訓練專用層
with torch.no_grad():                                               # 禁用梯度計算：節省記憶體，提高速度
    test_output = model(X_test)                                     # 測試集前向傳播：獲取預測
    test_loss = criterion(test_output, y_test)                      # 計算測試損失：評估測試性能
    mse = mean_squared_error(y_test.numpy(), test_output.numpy())   # 計算均方誤差：平均誤差平方
    r2 = r2_score(y_test.numpy(), test_output.numpy())              # 計算R²分數：解釋變異比例
    print(f"Test Loss: {test_loss.item():.4f}")                     # 印出測試損失：顯示測試結果
    print(f"Test MSE: {mse:.4f}")                                   # 印出測試MSE：同上
    print(f"Test R²: {r2:.4f}")                                     # 印出測試R²：同上

# 視覺化
plt.figure(figsize=(15, 10))  # 設定圖表大小：容納四個子圖

# 訓練損失曲線
plt.subplot(2, 2, 1)  # 第一個子圖
plt.plot(train_losses, label='Training Loss')  # 繪製損失曲線
plt.title('Training Loss Over Epochs')  # 標題
plt.xlabel('Epoch')  # x軸標籤
plt.ylabel('Loss')  # y軸標籤
plt.legend()  # 圖例

# 預測 vs 實際值散點圖
plt.subplot(2, 2, 2)  # 第二個子圖
plt.scatter(y_test.numpy(), test_output.numpy(), alpha=0.5)  # 散點圖
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Prediction')  # 完美預測線
plt.title('Predicted vs Actual Values')  # 標題
plt.xlabel('Actual Values')  # x軸標籤
plt.ylabel('Predicted Values')  # y軸標籤
plt.legend()  # 圖例

# 殘差圖 (Residual Plot)
residuals = y_test.numpy() - test_output.numpy()  # 計算殘差
plt.subplot(2, 2, 3)  # 第三個子圖
plt.scatter(test_output.numpy(), residuals, alpha=0.5)  # 殘差散點圖
plt.axhline(y=0, color='r', linestyle='--')  # 零線
plt.title('Residual Plot')  # 標題
plt.xlabel('Predicted Values')  # x軸標籤
plt.ylabel('Residuals')  # y軸標籤

# 預測誤差分佈直方圖
plt.subplot(2, 2, 4)  # 第四個子圖
plt.hist(residuals, bins=20, alpha=0.7, edgecolor='black')  # 直方圖
plt.title('Prediction Error Distribution')  # 標題
plt.xlabel('Error')  # x軸標籤
plt.ylabel('Frequency')  # y軸標籤

plt.tight_layout()  # 調整佈局
plt.savefig('diabetes_visualization.png')  # 保存圖片
plt.show()  # 顯示圖表

# 保存模型
torch.save(model.state_dict(), 'diabetes_model.pth')  # 保存模型狀態字典到文件：儲存訓練好的參數
