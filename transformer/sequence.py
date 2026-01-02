import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.nhead = nhead
        self.d_k = d_model // nhead
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(1)
        seq_len = q.size(0)
        
        # 修正：調整 transpose 以獲得正確形狀 (batch_size, nhead, seq_len, d_k)
        Q = self.w_q(q).view(seq_len, batch_size, self.nhead, self.d_k).transpose(0, 1).transpose(1, 2)
        K = self.w_k(k).view(seq_len, batch_size, self.nhead, self.d_k).transpose(0, 1).transpose(1, 2)
        V = self.w_v(v).view(seq_len, batch_size, self.nhead, self.d_k).transpose(0, 1).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        if mask is not None:
            # 修正：擴展遮罩以匹配 scores 的形狀 (batch_size, nhead, seq_len, seq_len)
            # 假設遮罩對於所有頭都相同，使用 (batch_size, 1, seq_len, seq_len)
            mask = mask.expand(batch_size, 1, seq_len, seq_len)
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        attention = F.softmax(scores, dim=-1)
        
        out = torch.matmul(attention, V)
        
        # 修正：調整 transpose 以獲得正確輸出形狀 (seq_len, batch_size, d_model)
        out = out.transpose(1, 2).contiguous().view(seq_len, batch_size, self.nhead * self.d_k)
        return self.fc_out(out)
    
class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, nhead)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forword = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        
    def forward(self, x, mask=None):
        attn_out = self.attention(x, x, x, mask)
        x = self.norm1(x + attn_out)
        
        ff_out = self.feed_forword(x)
        x = self.norm2(x + ff_out)
        
        return x
    
class HandmadeGPT(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(100, 1, d_model))
        
        self.layers = nn.ModuleList([TransformerBlock(d_model, nhead) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        seq_len, batch_size = x.size()
        
        mask = torch.tril(torch.ones((seq_len, seq_len))).to(x.device)
        
        mask = mask.unsqueeze(0).unsqueeze(0)
        
        out = self.embedding(x) + self.pos_emb[:seq_len, :, :]
        
        for layer in self.layers:
            out = layer(out, mask)
            
        return self.fc_out(out)

def generate_data(batch_size = 32):
    start = torch.randint(1, 20, (batch_size, 1))
    seq = (torch.arange(6) * 2.).unsqueeze(0)
    full_seq = (start + seq).t()
    
    return full_seq[:-1, :].long(), full_seq[1:, :].long()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HandmadeGPT(vocab_size=100, d_model=64, nhead=8, num_layers=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterition = nn.CrossEntropyLoss()

print("模型訓練中...")
for epoch in range(1001):
    x, y = generate_data()
    x, y = x.to(device), y.to(device)
    

    logits = model(x)
    
    loss = criterition(logits.view(-1, 100), y.reshape(-1))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
test_input = torch.tensor([[5], [7], [9], [11], [13]]).to(device) # 輸入 5, 7, 9, 11, 13
model.eval()
with torch.no_grad():
    prediction = model(test_input)
    # 取最後一個位置的輸出，看哪個數字機率最大
    result = prediction[-1, 0, :].argmax().item()
    print(f"\n輸入: 5, 7, 9, 11, 13")
    print(f"模型預測下一個數字是: {result}")