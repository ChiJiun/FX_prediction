import torch
import torch.nn as nn
import torch.optim as optim

# ==========================================
# 1. 定義模型 (Simple Transformer)
# ==========================================
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super(SimpleTransformer, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Transformer 模組
        self.transformer = nn.Transformer(
            d_model=d_model, 
            nhead=nhead, 
            num_encoder_layers=num_layers, 
            num_decoder_layers=num_layers,
            dropout=0.1
        )
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, trg, tgt_mask=None):
        # src, trg 維度均為 (Seq_Len, Batch)
        # 必須乘以 sqrt(d_model) 這是論文中的縮放約定
        src_emb = self.embedding(src) * (self.d_model ** 0.5)
        trg_emb = self.embedding(trg) * (self.d_model ** 0.5)
        
        # 進行 Transformer 計算
        out = self.transformer(src_emb, trg_emb, tgt_mask=tgt_mask)
        return self.fc_out(out)

# ==========================================
# 2. 數據生成函數 (數字排序)
# ==========================================
def generate_data(batch_size, length=5):
    # 隨機生成 1-10 的數字 (0 留給 <SOS>)
    src = torch.randint(1, 11, (batch_size, length)) 
    # 排序作為目標
    target = torch.sort(src, dim=1)[0]
    
    # Decoder 的輸入 (加上 <SOS> 並去掉最後一個，保持長度不變)
    sos = torch.zeros((batch_size, 1), dtype=torch.long)
    trg_input = torch.cat([sos, target[:, :-1]], dim=1)
    
    # 返回 (Seq, Batch) 格式
    return src.t(), trg_input.t(), target.t()

# ==========================================
# 3. 訓練模型
# ==========================================
# 參數設置
VOCAB_SIZE = 12  # 0: <SOS>, 1-10: 數字, 11: 冗餘
D_MODEL = 64
NHEAD = 8
NUM_LAYERS = 3
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleTransformer(VOCAB_SIZE, D_MODEL, NHEAD, NUM_LAYERS).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss()

print("開始訓練...")
model.train()
for epoch in range(2001):
    src, trg_input, trg_expected = generate_data(BATCH_SIZE)
    src, trg_input, trg_expected = src.to(device), trg_input.to(device), trg_expected.to(device)
    
    # 生成 Decoder 遮罩 (防止偷看未來答案)
    tgt_mask = nn.Transformer.generate_square_subsequent_mask(trg_input.size(0)).to(device)
    
    optimizer.zero_grad()
    output = model(src, trg_input, tgt_mask=tgt_mask)
    
    # 計算損失 (output: [Seq, Batch, Vocab] -> flatten)
    loss = criterion(output.view(-1, VOCAB_SIZE), trg_expected.reshape(-1))
    loss.backward()
    optimizer.step()
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# ==========================================
# 4. 推理 (測試模型是否學會排序)
# ==========================================
def evaluate(model, input_seq):
    model.eval()
    with torch.no_grad():
        # input_seq: (Seq_Len, 1)
        src = torch.tensor(input_seq).unsqueeze(1).to(device)
        # 初始 Decoder 輸入只有 <SOS>
        trg_input = torch.tensor([[0]]).to(device) 
        
        for _ in range(len(input_seq)):
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(trg_input.size(0)).to(device)
            out = model(src, trg_input, tgt_mask=tgt_mask)
            # 取最後一個時間步的預測值
            next_word = out.argmax(dim=-1)[-1, :].item()
            # 將預測出的字拼接到下一次的輸入中
            trg_input = torch.cat([trg_input, torch.tensor([[next_word]]).to(device)], dim=0)
            
        return trg_input.squeeze().tolist()[1:] # 去掉開頭的 <SOS>

# 測試
test_input = [8, 2, 5, 1, 9]
predicted = evaluate(model, test_input)
print(f"\n測試輸入: {test_input}")
print(f"模型排序結果: {predicted}")