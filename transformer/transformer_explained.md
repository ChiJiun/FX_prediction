# PyTorch Transformer 原始碼詳解

本文件詳細解釋 `torch.nn.modules.transformer.py` 中的各個類別。

---

## 目錄
1. [整體架構概覽](#整體架構概覽)
2. [Transformer 主類別](#transformer-主類別)
3. [TransformerEncoder](#transformerencoder)
4. [TransformerDecoder](#transformerdecoder)
5. [TransformerEncoderLayer](#transformerencoderlayer)
6. [TransformerDecoderLayer](#transformerdecoderlayer)
7. [輔助函數](#輔助函數)

---

## 整體架構概覽

```
┌─────────────────────────────────────────────────────────────┐
│                      Transformer                            │
│  ┌─────────────────────┐    ┌─────────────────────────┐    │
│  │  TransformerEncoder │    │  TransformerDecoder     │    │
│  │  ┌───────────────┐  │    │  ┌───────────────────┐  │    │
│  │  │EncoderLayer×N │  │    │  │ DecoderLayer×N    │  │    │
│  │  │ - Self-Attn   │  │    │  │ - Self-Attn       │  │    │
│  │  │ - FFN         │  │    │  │ - Cross-Attn      │  │    │
│  │  └───────────────┘  │    │  │ - FFN             │  │    │
│  └─────────────────────┘    │  └───────────────────┘  │    │
│                             └─────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## Transformer 主類別

### 類別定義

```python
class Transformer(Module):
```

這是完整的 Transformer 模型，包含 Encoder 和 Decoder。

### 初始化參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `d_model` | 512 | 模型維度（嵌入維度） |
| `nhead` | 8 | 多頭注意力的頭數 |
| `num_encoder_layers` | 6 | Encoder 層數 |
| `num_decoder_layers` | 6 | Decoder 層數 |
| `dim_feedforward` | 2048 | FFN 中間層維度 |
| `dropout` | 0.1 | Dropout 比率 |
| `activation` | relu | 激活函數 |
| `batch_first` | False | 輸入格式是否為 (Batch, Seq, Feature) |
| `norm_first` | False | 是否先做 LayerNorm（Pre-LN vs Post-LN） |

### `__init__` 方法詳解

```python
def __init__(self, d_model=512, nhead=8, ...):
    # factory_kwargs 用於指定張量創建時的 device 和 dtype
    factory_kwargs = {"device": device, "dtype": dtype}
    super().__init__()
    
    # ===== 建立 Encoder =====
    if custom_encoder is not None:
        # 允許使用自定義 Encoder
        self.encoder = custom_encoder
    else:
        # 創建單個 EncoderLayer 作為模板
        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, ...
        )
        # 最後的 LayerNorm
        encoder_norm = LayerNorm(d_model, eps=layer_norm_eps, ...)
        # 堆疊 N 個 EncoderLayer
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )
    
    # ===== 建立 Decoder（結構類似）=====
    # ...
    
    # 初始化參數（Xavier 初始化）
    self._reset_parameters()
```

### `forward` 方法詳解

```python
def forward(self, src, tgt, src_mask=None, tgt_mask=None, ...):
    """
    參數:
        src: 源序列（給 Encoder）
             維度: (S, N, E) 或 (N, S, E) if batch_first=True
        tgt: 目標序列（給 Decoder）
             維度: (T, N, E) 或 (N, T, E) if batch_first=True
        src_mask: Encoder 自注意力遮罩 (S, S)
        tgt_mask: Decoder 自注意力遮罩 (T, T)（通常是因果遮罩）
        memory_mask: Cross-Attention 遮罩 (T, S)
        *_key_padding_mask: 填充位置的遮罩
        *_is_causal: 是否為因果遮罩的提示
    """
    # 檢查 batch size 一致性
    is_batched = src.dim() == 3
    if not self.batch_first and src.size(1) != tgt.size(1):
        raise RuntimeError("batch size 不一致")
    
    # 檢查特徵維度
    if src.size(-1) != self.d_model:
        raise RuntimeError("特徵維度必須等於 d_model")
    
    # 步驟 1: Encoder 處理源序列
    # memory 是 Encoder 的輸出，維度與 src 相同
    memory = self.encoder(
        src,
        mask=src_mask,
        src_key_padding_mask=src_key_padding_mask,
        is_causal=src_is_causal,
    )
    
    # 步驟 2: Decoder 處理目標序列 + Encoder 輸出
    output = self.decoder(
        tgt,
        memory,  # Encoder 輸出作為 Cross-Attention 的 Key/Value
        tgt_mask=tgt_mask,
        memory_mask=memory_mask,
        ...
    )
    
    return output  # 維度: (T, N, E)
```

### `generate_square_subsequent_mask` 靜態方法

```python
@staticmethod
def generate_square_subsequent_mask(sz, device=None, dtype=None):
    """
    生成因果遮罩（下三角矩陣）
    
    例如 sz=5 時生成:
    [[  0, -inf, -inf, -inf, -inf],
     [  0,    0, -inf, -inf, -inf],
     [  0,    0,    0, -inf, -inf],
     [  0,    0,    0,    0, -inf],
     [  0,    0,    0,    0,    0]]
    
    0 表示可以注意，-inf 經過 softmax 後變成 0（不注意）
    這確保位置 i 只能看到位置 0 到 i
    """
    return torch.triu(
        torch.full((sz, sz), float("-inf"), ...),
        diagonal=1,  # 主對角線以上填 -inf
    )
```

---

## TransformerEncoder

### 類別定義

```python
class TransformerEncoder(Module):
    """
    堆疊 N 個 TransformerEncoderLayer
    """
```

### 初始化

```python
def __init__(self, encoder_layer, num_layers, norm=None, enable_nested_tensor=True):
    # 複製 encoder_layer N 次
    self.layers = _get_clones(encoder_layer, num_layers)
    self.num_layers = num_layers
    self.norm = norm  # 最後的 LayerNorm（可選）
    
    # Nested Tensor 優化選項（處理變長序列更高效）
    self.enable_nested_tensor = enable_nested_tensor
```

### `forward` 方法

```python
def forward(self, src, mask=None, src_key_padding_mask=None, is_causal=None):
    output = src
    
    # ===== 快速路徑檢查（用於性能優化）=====
    # 檢查是否符合使用優化實現的條件
    # 例如：batch_first=True、推理模式、無梯度等
    
    # ===== 逐層處理 =====
    for mod in self.layers:
        output = mod(
            output,
            src_mask=mask,
            is_causal=is_causal,
            src_key_padding_mask=src_key_padding_mask,
        )
    
    # ===== 最後的 LayerNorm =====
    if self.norm is not None:
        output = self.norm(output)
    
    return output
```

---

## TransformerDecoder

### 類別定義

```python
class TransformerDecoder(Module):
    """
    堆疊 N 個 TransformerDecoderLayer
    """
```

### `forward` 方法

```python
def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, ...):
    """
    參數:
        tgt: 目標序列
        memory: Encoder 的輸出（用於 Cross-Attention）
    """
    output = tgt
    
    # 檢測是否為因果遮罩
    seq_len = _get_seq_len(tgt, self.layers[0].self_attn.batch_first)
    tgt_is_causal = _detect_is_causal_mask(tgt_mask, tgt_is_causal, seq_len)
    
    # 逐層處理
    for mod in self.layers:
        output = mod(
            output,
            memory,  # 每層都接收 Encoder 輸出
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            ...
        )
    
    if self.norm is not None:
        output = self.norm(output)
    
    return output
```

---

## TransformerEncoderLayer

### 架構圖

```
輸入 x
    │
    ▼
┌───────────────┐
│  Self-Attn    │◄── Q, K, V 都來自 x
└───────────────┘
    │
    ▼ (+殘差連接)
┌───────────────┐
│  LayerNorm    │
└───────────────┘
    │
    ▼
┌───────────────┐
│    FFN        │ Linear → ReLU → Linear
└───────────────┘
    │
    ▼ (+殘差連接)
┌───────────────┐
│  LayerNorm    │
└───────────────┘
    │
    ▼
輸出
```

### 初始化

```python
def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, ...):
    # ===== Self-Attention =====
    self.self_attn = MultiheadAttention(
        d_model,      # 嵌入維度
        nhead,        # 頭數
        dropout=dropout,
        batch_first=batch_first,
    )
    
    # ===== Feed Forward Network (FFN) =====
    # FFN: Linear(d_model → dim_feedforward) → ReLU → Linear(dim_feedforward → d_model)
    self.linear1 = Linear(d_model, dim_feedforward)  # 擴展
    self.dropout = Dropout(dropout)
    self.linear2 = Linear(dim_feedforward, d_model)  # 壓縮回來
    
    # ===== Layer Normalization =====
    self.norm1 = LayerNorm(d_model)  # Self-Attn 後
    self.norm2 = LayerNorm(d_model)  # FFN 後
    
    # ===== Dropout =====
    self.dropout1 = Dropout(dropout)  # Self-Attn 後
    self.dropout2 = Dropout(dropout)  # FFN 後
    
    # norm_first: Pre-LN (True) vs Post-LN (False)
    self.norm_first = norm_first
```

### `forward` 方法

```python
def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
    x = src
    
    if self.norm_first:
        # ===== Pre-LN 架構 =====
        # 先 Norm，再 Attention，再殘差
        x = x + self._sa_block(self.norm1(x), src_mask, ...)
        x = x + self._ff_block(self.norm2(x))
    else:
        # ===== Post-LN 架構（原始 Transformer）=====
        # 先 Attention，再殘差，再 Norm
        x = self.norm1(x + self._sa_block(x, src_mask, ...))
        x = self.norm2(x + self._ff_block(x))
    
    return x
```

### 子模組方法

```python
def _sa_block(self, x, attn_mask, key_padding_mask, is_causal=False):
    """Self-Attention 區塊"""
    # Q=K=V=x（自注意力）
    x = self.self_attn(
        x, x, x,  # Query, Key, Value 都是 x
        attn_mask=attn_mask,
        key_padding_mask=key_padding_mask,
        is_causal=is_causal,
        need_weights=False,  # 不返回注意力權重
    )[0]  # [0] 是輸出，[1] 是注意力權重
    return self.dropout1(x)

def _ff_block(self, x):
    """Feed Forward Network 區塊"""
    # Linear → Activation → Dropout → Linear → Dropout
    x = self.linear2(self.dropout(self.activation(self.linear1(x))))
    return self.dropout2(x)
```

---

## TransformerDecoderLayer

### 架構圖

```
目標輸入 tgt          Encoder 輸出 memory
    │                        │
    ▼                        │
┌───────────────┐            │
│ Masked        │            │
│ Self-Attn     │◄── Q,K,V 都來自 tgt
└───────────────┘            │
    │                        │
    ▼ (+殘差+Norm)           │
┌───────────────┐            │
│ Cross-Attn    │◄── Q 來自 tgt, K,V 來自 memory
└───────────────┘            │
    │                        │
    ▼ (+殘差+Norm)           │
┌───────────────┐
│    FFN        │
└───────────────┘
    │
    ▼ (+殘差+Norm)
輸出
```

### 初始化

```python
def __init__(self, d_model, nhead, ...):
    # ===== Masked Self-Attention（看自己）=====
    self.self_attn = MultiheadAttention(d_model, nhead, ...)
    
    # ===== Cross-Attention（看 Encoder 輸出）=====
    self.multihead_attn = MultiheadAttention(d_model, nhead, ...)
    
    # ===== FFN =====
    self.linear1 = Linear(d_model, dim_feedforward)
    self.linear2 = Linear(dim_feedforward, d_model)
    
    # ===== LayerNorm（3個：Self-Attn, Cross-Attn, FFN 各一個）=====
    self.norm1 = LayerNorm(d_model)
    self.norm2 = LayerNorm(d_model)
    self.norm3 = LayerNorm(d_model)
```

### `forward` 方法

```python
def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, ...):
    x = tgt
    
    if self.norm_first:
        # Pre-LN
        x = x + self._sa_block(self.norm1(x), tgt_mask, ...)     # Self-Attn
        x = x + self._mha_block(self.norm2(x), memory, ...)      # Cross-Attn
        x = x + self._ff_block(self.norm3(x))                    # FFN
    else:
        # Post-LN
        x = self.norm1(x + self._sa_block(x, tgt_mask, ...))
        x = self.norm2(x + self._mha_block(x, memory, ...))
        x = self.norm3(x + self._ff_block(x))
    
    return x
```

### Cross-Attention 區塊

```python
def _mha_block(self, x, mem, attn_mask, key_padding_mask, is_causal=False):
    """Cross-Attention 區塊"""
    x = self.multihead_attn(
        x,    # Query: 來自 Decoder
        mem,  # Key: 來自 Encoder
        mem,  # Value: 來自 Encoder
        attn_mask=attn_mask,
        key_padding_mask=key_padding_mask,
        is_causal=is_causal,
        need_weights=False,
    )[0]
    return self.dropout2(x)
```

---

## 輔助函數

### `_get_clones`

```python
def _get_clones(module, N):
    """複製模組 N 次，返回 ModuleList"""
    return ModuleList([copy.deepcopy(module) for i in range(N)])
```

### `_get_activation_fn`

```python
def _get_activation_fn(activation: str):
    """根據字串返回激活函數"""
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError(f"不支援的激活函數: {activation}")
```

### `_detect_is_causal_mask`

```python
def _detect_is_causal_mask(mask, is_causal=None, size=None):
    """
    檢測給定的遮罩是否為因果遮罩
    
    如果 is_causal 已指定，直接返回
    否則，生成一個標準因果遮罩並比較
    """
    if is_causal is None and mask is not None:
        # 生成標準因果遮罩
        causal_comparison = _generate_square_subsequent_mask(sz, ...)
        # 比較是否相同
        make_causal = bool((mask == causal_comparison).all())
    return make_causal
```

### `_generate_square_subsequent_mask`

```python
def _generate_square_subsequent_mask(sz, device=None, dtype=None):
    """
    生成因果遮罩
    
    torch.triu: 取上三角矩陣
    diagonal=1: 從主對角線上方一格開始
    
    結果：下三角為 0（可注意），上三角為 -inf（不可注意）
    """
    return torch.triu(
        torch.full((sz, sz), float("-inf"), dtype=dtype, device=device),
        diagonal=1,
    )
```

---

## 資料流總結

```
輸入序列 src (S, N, E)
         │
         ▼
┌─────────────────────────────────────┐
│           ENCODER                   │
│  ┌─────────────────────────────┐   │
│  │ EncoderLayer × N            │   │
│  │  • Self-Attention(src, src) │   │
│  │  • FFN                      │   │
│  └─────────────────────────────┘   │
│                │                    │
│                ▼                    │
│        memory (S, N, E)            │
└─────────────────────────────────────┘
                 │
                 │ Cross-Attention 的 K, V
                 ▼
┌─────────────────────────────────────┐
│           DECODER                   │
│  目標序列 tgt (T, N, E)            │
│         │                           │
│  ┌─────────────────────────────┐   │
│  │ DecoderLayer × N            │   │
│  │  • Masked Self-Attn(tgt)    │   │
│  │  • Cross-Attn(tgt, memory)  │   │
│  │  • FFN                      │   │
│  └─────────────────────────────┘   │
│                │                    │
│                ▼                    │
│        output (T, N, E)            │
└─────────────────────────────────────┘
```

---

## 重要概念

### 1. Pre-LN vs Post-LN

| | Post-LN (原始) | Pre-LN |
|---|---|---|
| 順序 | Attn → Add → Norm | Norm → Attn → Add |
| 訓練穩定性 | 較差 | 較好 |
| 參數 | `norm_first=False` | `norm_first=True` |

### 2. 遮罩類型

| 遮罩 | 用途 | 維度 |
|------|------|------|
| `src_mask` | Encoder 自注意力 | (S, S) |
| `tgt_mask` | Decoder 自注意力（因果） | (T, T) |
| `memory_mask` | Cross-Attention | (T, S) |
| `*_key_padding_mask` | 標記填充位置 | (N, S) 或 (N, T) |

### 3. 因果遮罩

確保位置 i 只能注意位置 0 到 i-1，用於自回歸生成：

```
位置:    0    1    2    3    4
    0 [  ✓   ✗   ✗   ✗   ✗  ]
    1 [  ✓   ✓   ✗   ✗   ✗  ]
    2 [  ✓   ✓   ✓   ✗   ✗  ]
    3 [  ✓   ✓   ✓   ✓   ✗  ]
    4 [  ✓   ✓   ✓   ✓   ✓  ]
```
