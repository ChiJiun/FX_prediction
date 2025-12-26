from diabetes_model import Simple3LayerNN 
import torch
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

# 載入數據（從 diabetes.py 複製）
diabetes = load_diabetes()  # 載入糖尿病數據集：獲取包含特徵和目標的數據
X = diabetes.data           # 獲取特徵數據：X是(442, 10)的特徵矩陣
y = diabetes.target         # 獲取目標變數：y是(442,)的目標向量
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 分割訓練和測試集，測試集佔20%：確保重現性，測試模型泛化

# 將 Numpy 陣列轉換為 PyTorch 張量
X_train = torch.tensor(X_train, dtype=torch.float32)                # 將訓練特徵轉換為PyTorch張量：PyTorch需要張量格式進行計算
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)    # 將訓練目標轉換為張量並調整形狀：從(353,)變為(353, 1)，匹配輸出
X_test = torch.tensor(X_test, dtype=torch.float32)                  # 將測試特徵轉換為PyTorch張量：同上
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)      # 將測試目標轉換為張量並調整形狀：同上

# 實例化模型
model = Simple3LayerNN(input_dim=X_train.shape[1])      # 實例化模型，輸入維度為訓練特徵數：建立模型實例

# 載入模型並進行預測
model.load_state_dict(torch.load('diabetes_model.pth'))  # 載入保存的模型參數
model.eval()  # 設定為評估模式

# 示例預測：使用測試數據的第一個樣本
with torch.no_grad():
    sample_input = X_test[0].unsqueeze(0)  # 取第一個測試樣本，增加批次維度
    prediction = model(sample_input)  # 進行預測
    print(f"Sample Prediction: {prediction.item():.4f}, Actual: {y_test[0].item():.4f}")  # 印出預測和實際值
