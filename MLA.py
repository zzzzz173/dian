import torch
import torch.nn as nn
import torch.nn.functional as F


class MLAAttention(nn.Module):
    def __init__(
            self,
            dim,  # 输入维度
            num_heads=8,  # 注意力头数
            latent_dim=64,  # 潜在空间维度
            dropout=0.1
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Q, K, V 投影矩阵
        self.q_proj = nn.Linear(dim, dim)

        # 低秩分解的K和V投影
        self.k_latent = nn.Linear(dim, num_heads * latent_dim)
        self.v_latent = nn.Linear(dim, num_heads * latent_dim)

        # 从潜在空间映射回原始空间
        self.k_proj = nn.Linear(latent_dim, self.head_dim)
        self.v_proj = nn.Linear(latent_dim, self.head_dim)

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        batch_size, seq_len, dim = x.shape

        # 生成Q并分头
        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        q = q * self.scale

        # 生成低秩K和V
        k_latent = self.k_latent(x).reshape(batch_size, seq_len, self.num_heads, -1)
        v_latent = self.v_latent(x).reshape(batch_size, seq_len, self.num_heads, -1)

        # 从潜在空间映射到原始空间
        k = self.k_proj(k_latent)
        v = self.v_proj(v_latent)

        # 调整维度顺序为 (batch_size, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 计算注意力分数
        attn = torch.matmul(q, k.transpose(-2, -1))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # 应用注意力权重
        out = torch.matmul(attn, v)

        # 重塑并投影输出
        out = out.transpose(1, 2).reshape(batch_size, seq_len, dim)
        out = self.out_proj(out)

        return out

batch_size = 32
seq_len = 512
dim = 768

# 初始化MLA
mla = MLAAttention(
    dim=dim,
    num_heads=8,
    latent_dim=64,
    dropout=0.1
)

# 创建输入张量
x = torch.randn(batch_size, seq_len, dim)

# 前向传播
output = mla(x)
print(output.shape)