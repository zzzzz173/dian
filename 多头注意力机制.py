import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        # 确保d_model可以被num_heads整除
        assert d_model % num_heads == 0

        self.d_model = d_model  # 模型维度
        self.num_heads = num_heads  # 注意力头数
        self.d_k = d_model // num_heads  # 每个头的维度

        # 定义线性变换层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V):
        # 计算注意力权重
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)

        # 应用注意力权重到V上
        output = torch.matmul(attention_weights, V)
        return output, attention_weights

    def forward(self, Q, K, V):
        batch_size = Q.size(0)

        # 线性变换
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)

        # 将张量分割成多个头
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 计算注意力
        output, attention_weights = self.scaled_dot_product_attention(Q, K, V)

        # 重组多头的输出
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 最后的线性变换
        output = self.W_o(output)

        return output, attention_weights


class MultiQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)  # num_heads个Q
        self.W_k = nn.Linear(d_model, self.d_k)  # 单个K
        self.W_v = nn.Linear(d_model, self.d_k)  # 单个V
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V):
        batch_size, seq_len, _ = Q.size()

        # Q维度: [batch_size, seq_len, num_heads, d_k]
        Q = self.W_q(Q).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # K和V只有一个头: [batch_size, seq_len, d_k]
        K = self.W_k(K)
        V = self.W_v(V)

        # 扩展K和V到所有头
        # [batch_size, seq_len, d_k] -> [batch_size, num_heads, seq_len, d_k]
        K = K.unsqueeze(1).expand(batch_size, self.num_heads, seq_len, self.d_k)
        V = V.unsqueeze(1).expand(batch_size, self.num_heads, seq_len, self.d_k)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)

        # 计算输出
        output = torch.matmul(attention_weights, V)

        # 重组输出
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(output)

        return output, attention_weights


class GroupQueryAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8, num_groups=2, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        assert num_heads % num_groups == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.heads_per_group = num_heads // num_groups
        self.d_k = d_model // num_heads

        # GQA：Q是多头的，但K和V是分组的
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model // (num_heads // num_groups))
        self.W_v = nn.Linear(d_model, d_model // (num_heads // num_groups))
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch_size, seq_len, _ = q.size()

        # Q保持多头
        Q = self.W_q(q).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # K和V是分组的
        K = self.W_k(k).view(batch_size, seq_len, self.num_groups, self.d_k)
        V = self.W_v(v).view(batch_size, seq_len, self.num_groups, self.d_k)

        # 扩展K和V到对应的头
        K = K.unsqueeze(2).expand(-1, -1, self.heads_per_group, -1, -1)
        V = V.unsqueeze(2).expand(-1, -1, self.heads_per_group, -1, -1)
        K = K.contiguous().view(batch_size, self.num_heads, seq_len, self.d_k)
        V = V.contiguous().view(batch_size, self.num_heads, seq_len, self.d_k)

        # 计算注意力权重
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(context)

        return output, attn


# 测试代码
def test_attention_mechanisms():
    # 设置参数
    batch_size = 2
    seq_length = 10
    d_model = 512
    num_heads = 8
    num_groups = 2  # GQA的组数

    # 创建随机输入
    x = torch.randn(batch_size, seq_length, d_model)

    # 初始化不同的注意力机制
    mha = MultiHeadAttention(d_model, num_heads)
    mqa = MultiQueryAttention(d_model, num_heads)
    gqa = GroupQueryAttention(d_model, num_heads, num_groups)

    # 测试各种注意力机制
    mha_output, mha_attn = mha(x, x, x)
    mqa_output, mqa_attn = mqa(x, x, x)
    gqa_output, gqa_attn = gqa(x, x, x)

    print("输入张量形状:", x.shape)
    print("\nMHA输出形状:", mha_output.shape)
    print("MHA注意力权重形状:", mha_attn.shape)
    print("\nMQA输出形状:", mqa_output.shape)
    print("MQA注意力权重形状:", mqa_attn.shape)
    print("\nGQA输出形状:", gqa_output.shape)
    print("GQA注意力权重形状:", gqa_attn.shape)

    # 详细分析注意力权重差异
    print("\n=== 注意力权重详细分析 ===")

    # 1. 计算平均差异
    mha_mqa_diff = torch.mean(torch.abs(mha_attn - mqa_attn)).item()
    mha_gqa_diff = torch.mean(torch.abs(mha_attn - gqa_attn)).item()
    mqa_gqa_diff = torch.mean(torch.abs(mqa_attn - gqa_attn)).item()

    print("\n1. 平均权重差异:")
    print(f"MHA vs MQA: {mha_mqa_diff:.4f}")
    print(f"MHA vs GQA: {mha_gqa_diff:.4f}")
    print(f"MQA vs GQA: {mqa_gqa_diff:.4f}")

    # 2. 计算最大差异
    mha_mqa_max = torch.max(torch.abs(mha_attn - mqa_attn)).item()
    mha_gqa_max = torch.max(torch.abs(mha_attn - gqa_attn)).item()
    mqa_gqa_max = torch.max(torch.abs(mqa_attn - gqa_attn)).item()

    print("\n2. 最大权重差异:")
    print(f"MHA vs MQA: {mha_mqa_max:.4f}")
    print(f"MHA vs GQA: {mha_gqa_max:.4f}")
    print(f"MQA vs GQA: {mqa_gqa_max:.4f}")

    # 3. 计算权重分布统计
    print("\n3. 权重分布统计:")
    print("MHA权重统计:")
    print(f"  均值: {torch.mean(mha_attn):.4f}")
    print(f"  标准差: {torch.std(mha_attn):.4f}")
    print(f"  最小值: {torch.min(mha_attn):.4f}")
    print(f"  最大值: {torch.max(mha_attn):.4f}")

    print("\nMQA权重统计:")
    print(f"  均值: {torch.mean(mqa_attn):.4f}")
    print(f"  标准差: {torch.std(mqa_attn):.4f}")
    print(f"  最小值: {torch.min(mqa_attn):.4f}")
    print(f"  最大值: {torch.max(mqa_attn):.4f}")

    print("\nGQA权重统计:")
    print(f"  均值: {torch.mean(gqa_attn):.4f}")
    print(f"  标准差: {torch.std(gqa_attn):.4f}")
    print(f"  最小值: {torch.min(gqa_attn):.4f}")
    print(f"  最大值: {torch.max(gqa_attn):.4f}")

    # 4. 分析第一个样本第一个头的注意力权重
    print("\n4. 第一个样本第一个头的注意力权重示例:")
    print("MHA:", mha_attn[0, 0])
    print("MQA:", mqa_attn[0, 0])
    print("GQA:", gqa_attn[0, 0])


if __name__ == "__main__":
    test_attention_mechanisms()