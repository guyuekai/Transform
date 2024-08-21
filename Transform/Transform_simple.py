import os
import requests
import math
import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F

# 超参数
batch_size = 6
context_length = 16  # 一个句子的单词数量
d_model = 512  # 每个词扩充的维度
num_blocks = 6  # Transform块的个数
num_heads = 4  # 多头注意力的参数
learning_rate = 1e-3  # 0.001
dropout = 0.1  # Dropout rate
max_iters = 10000  # 训练迭代总数 <- 将其更改为较小的数字以进行测试
eval_interval = 50  # 多久评估一次
eval_iters = 20  # 评估的平均迭代次数
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if it's available.
TORCH_SEED = 1337
torch.manual_seed(TORCH_SEED)

# Load training data
if not os.path.exists('sales_textbook.txt'):
    url = 'https://huggingface.co/datasets/goendalf666/sales-textbook_for_convincing_and_selling/raw/main/sales_textbook.txt'
    with open('sales_textbook.txt', 'w') as f:
        f.write(requests.get(url).text)

with open('sales_textbook.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Using TikToken (Same as GPT3) to tokenize the source text
encoding = tiktoken.get_encoding("cl100k_base")
tokenized_text = encoding.encode(text)
max_token_value = max(tokenized_text) + 1  # the maximum value of the tokenized numbers
tokenized_text = torch.tensor(tokenized_text, dtype=torch.long, device=device)  # 转化为张量

# Split train and validation
split_idx = int(len(tokenized_text) * 0.9)
train_data = tokenized_text[:split_idx]
val_data = tokenized_text[split_idx:]


# Define Feed Forward Network
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.ffn = nn.Sequential(
            nn.Linear(in_features=self.d_model, out_features=self.d_model * 4),
            nn.ReLU(),
            nn.Linear(in_features=self.d_model * 4, out_features=self.d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.ffn(x)


# 点积注意力分数
class Attention(nn.Module):
    def __init__(self, head_size: int):
        super().__init__()
        self.d_model = d_model
        # 每个头的向量长度
        self.head_size = head_size
        self.context_length = context_length
        self.dropout = dropout

        self.key_layer = nn.Linear(in_features=self.d_model, out_features=self.head_size, bias=False)
        self.query_layer = nn.Linear(in_features=self.d_model, out_features=self.head_size, bias=False)
        self.value_layer = nn.Linear(in_features=self.d_model, out_features=self.head_size, bias=False)
        # 生成一个下三角矩阵，用作掩码
        self.register_buffer('tril', torch.tril(
            torch.ones((self.context_length, self.context_length))))  # Lower triangular mask
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x):
        B, T, C = x.shape  # Batch size, Time steps(current context_length), Channels(dimensions)
        assert T <= self.context_length
        assert C == self.d_model
        # 得到q,k,v
        q = self.query_layer(x)
        k = self.key_layer(x)
        v = self.value_layer(x)

        # 点积注意力: Q @ K^T / sqrt(d_k)
        weights = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # 添加掩码信息，我们只得到之前的信息
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        # 输出为概率
        weights = F.softmax(input=weights, dim=-1)
        weights = self.dropout_layer(weights)

        # Apply dot product attention: weights @ V
        out = weights @ v
        return out

# 多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, head_size: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.d_model = d_model
        self.context_length = context_length
        self.dropout = dropout

        # num_heads为多少就有多少个点积注意力分数模块，每个输入都会被Attention里面的权重分割，进行计算，实现了多头注意力机制的计算
        # 假设num_heads为4，每个heads里面就有四个Attention块
        self.heads = nn.ModuleList([Attention(head_size=self.head_size) for _ in range(self.num_heads)])
        self.projection_layer = nn.Linear(in_features=self.d_model, out_features=self.d_model)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        # 合并多头注意力的输出
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # 最后经过一个Wo，确保形状
        out = self.projection_layer(out)
        out = self.dropout_layer(out)
        return out

# encoder模块
class TransformerBlock(nn.Module):

    def __init__(self, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.context_length = context_length
        self.head_size = d_model // num_heads  # head size should be divisible by d_model
        self.num_heads = num_heads
        self.dropout = dropout

        self.multi_head_attention_layer = MultiHeadAttention(head_size=self.head_size)
        self.feed_forward_layer = FeedForward()
        self.layer_norm_1 = nn.LayerNorm(normalized_shape=self.d_model)
        self.layer_norm_2 = nn.LayerNorm(normalized_shape=self.d_model)

    def forward(self, x):
        # 这里先进行层归一化
        x = x + self.multi_head_attention_layer(self.layer_norm_1(x))  # Residual connection
        x = x + self.feed_forward_layer(self.layer_norm_2(x))  # Residual connection
        return x


class TransformerLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = d_model
        self.context_length = context_length
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.max_token_value = max_token_value
        # 将每个token进行向量化
        self.token_embedding_lookup_table = nn.Embedding(num_embeddings=self.max_token_value + 1, embedding_dim=self.d_model)

        # 组合多个encoder块
        # 与原始论文不同，这里我们在所有块之后添加最终层范数
        self.transformer_blocks = nn.Sequential(*(
                [TransformerBlock(num_heads=self.num_heads) for _ in range(self.num_blocks)] +
                [nn.LayerNorm(self.d_model)]
        ))
        # 输出层
        self.language_model_out_linear_layer = nn.Linear(in_features=self.d_model, out_features=self.max_token_value)

    def forward(self, idx, targets=None):
        # idx是（B ， T）
        # B：batch_size , T:context_length
        B, T = idx.shape
        """
        # 生成位置嵌入表
        # 采用与原始 Transformer 论文相同的方法（正弦和余弦函数）
        """
        position_encoding_lookup_table = torch.zeros(self.context_length, self.d_model)
        position = torch.arange(0, self.context_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        position_encoding_lookup_table[:, 0::2] = torch.sin(position * div_term)
        position_encoding_lookup_table[:, 1::2] = torch.cos(position * div_term)
        # change position_encoding_lookup_table from (context_length, d_model) to (T, d_model)
        position_embedding = position_encoding_lookup_table[:T, :].to(device)
        # 先按照idx获取数据 ， 然后给数据加上位置信息
        # 如何获取数据见jupyter
        x = self.token_embedding_lookup_table(idx) + position_embedding
        x = self.transformer_blocks(x)
        # “logits”是应用 softmax 之前模型的输出值
        logits = self.language_model_out_linear_layer(x)
        # targets不为空时计算交叉熵误差
        if targets is not None:
            B, T, C = logits.shape
            # F.cross_entropy 函数期望输入的形状为 (N, C)，因此合并前两个维度
            logits_reshaped = logits.view(B * T, C)
            # 将 targets 的形状展平成 (B * T)，使其与 logits_reshaped 对应
            targets_reshaped = targets.view(B * T)
            loss = F.cross_entropy(input=logits_reshaped, target=targets_reshaped)
        else:
            loss = None
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx 是当前上下文中的索引 (B,T) 数组
        for _ in range(max_new_tokens):
            # 将 idx 裁剪为位置嵌入表的最大大小 , 获取数据一个大小为idx的数据
            idx_crop = idx[:, -self.context_length:]
            # 调用forward方法，输出结果
            logits, loss = self(idx_crop)
            # 从 logits 获取最后一个时间步长，是每个序列的最后一个时间步的所有特征
            logits_last_timestep = logits[:, -1, :]
            # 应用softmax来获取概率
            probs = F.softmax(input=logits_last_timestep, dim=-1)
            # 从模型的输出概率中选取下一个生成的词或符号
            # 使用 torch.multinomial 从这个分布中采样，模型可以生成下一个符号，这个过程可以重复多次，以生成整个序列
            idx_next = torch.multinomial(input=probs, num_samples=1)
            # 将预测出的结果和已有的结果拼接
            # 模型通过循环生成一个一个的元素（词、音符等）。每次生成新元素后，将其添加到已有序列中，逐步扩展序列，直到达到所需的长度或生成终止符号
            # 拼接后的 idx 张量的形状将变为 (B, T + 1)，表示序列长度增加了一个时间步
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# 初始化模型
model = TransformerLanguageModel()
model = model.to(device)


# 获取词嵌入向量
def get_batch(split: str):
    data = train_data if split == 'train' else val_data
    # 随机生成索引batch_size个索引，用于获取batch_size个长context_length的序列
    idxs = torch.randint(low=0, high=len(data) - context_length, size=(batch_size,))
    # 获取x  y
    x = torch.stack([data[idx:idx + context_length] for idx in idxs]).to(device)
    y = torch.stack([data[idx + 1:idx + context_length + 1] for idx in idxs]).to(device)
    return x, y


# Calculate loss
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'valid']:
        # 创建了一个长度为 eval_iters 的全零张量 losses，用于存储在评估过程中每次迭代的损失值
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x_batch, y_batch = get_batch(split)
            logits, loss = model(x_batch, y_batch)
            # loss.item() 方法将单一元素的张量转换为一个 Python 标量（浮点数或整数）
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# 使用 AdamW 优化器
optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)
tracked_losses = list()
for step in range(max_iters):
    if step % eval_iters == 0 or step == max_iters - 1:
        losses = estimate_loss()
        tracked_losses.append(losses)
        print('Step:', step, 'Training Loss:', round(losses['train'].item(), 3), 'Validation Loss:',
              round(losses['valid'].item(), 3))

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# 保存模型参数
torch.save(model.state_dict(), 'model-ckpt.pt')

# 生成预测
model.eval()
start = 'The salesperson'
start_ids = encoding.encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
y = model.generate(x, max_new_tokens=100)
print('---------------')
print(encoding.decode(y[0].tolist()))
print('---------------')