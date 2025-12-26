import torch                                                # 匯入PyTorch庫
import torch.nn as nn                                       # 匯入神經網路模組

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

