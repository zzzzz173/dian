import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_key_value_heads=None, rope_base=10000):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads or num_heads
        self.head_dim = d_model // num_heads

        # 投影矩阵
        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, d_model, bias=False)

        # 用于GQA的重复映射矩阵
        if self.num_key_value_heads != self.num_heads:
            self.kv_repeat_matrix = self._create_repeat_matrix()

        # RoPE相关参数
        self.rope_base = rope_base
        self.register_buffer(
            "rope_frequency",
            1.0 / (rope_base ** (torch.arange(0, self.head_dim // 2).float() / (self.head_dim // 2)))
        )

    def _create_repeat_matrix(self):
        """创建用于GQA的重复映射矩阵"""
        repeats = self.num_heads // self.num_key_value_heads
        return torch.repeat_interleave(
            torch.arange(self.num_key_value_heads),
            repeats
        )

    def apply_rotary_embedding(self, x, position_ids):
        """应用RoPE位置编码"""
        batch_size, seq_length, num_heads, head_dim = x.shape

        # 调整position_ids的维度
        position_ids = position_ids.unsqueeze(-1).unsqueeze(-1)  # [batch, seq_len, 1, 1]

        # 计算sin和cos
        theta = position_ids * self.rope_frequency  # [batch, seq_len, 1, head_dim//2]
        sin = torch.sin(theta).unsqueeze(-2)  # [batch, seq_len, 1, 1, head_dim//2]
        cos = torch.cos(theta).unsqueeze(-2)  # [batch, seq_len, 1, 1, head_dim//2]

        # 调整x的形状以进行旋转
        x_reshape = x.view(batch_size, seq_length, num_heads, 2, -1)  # [batch, seq_len, num_heads, 2, head_dim//2]

        # 分离实部和虚部
        x1, x2 = x_reshape.unbind(dim=-2)  # 两个 [batch, seq_len, num_heads, head_dim//2]

        # 进行旋转操作
        rotated_x = torch.stack([
            x1 * cos.squeeze(-2) - x2 * sin.squeeze(-2),
            x2 * cos.squeeze(-2) + x1 * sin.squeeze(-2)
        ], dim=-2)  # [batch, seq_len, num_heads, 2, head_dim//2]

        return rotated_x.flatten(-2)  # [batch, seq_len, num_heads, head_dim]

    def forward(self, hidden_states, position_ids=None, attention_mask=None, past_key_value=None):
        batch_size, seq_length = hidden_states.shape[:2]

        # 投影Q、K、V
        query_states = self.q_proj(hidden_states).view(
            batch_size, seq_length, self.num_heads, self.head_dim
        )
        key_states = self.k_proj(hidden_states).view(
            batch_size, seq_length, self.num_key_value_heads, self.head_dim
        )
        value_states = self.v_proj(hidden_states).view(
            batch_size, seq_length, self.num_key_value_heads, self.head_dim
        )

        # 应用RoPE
        if position_ids is not None:
            query_states = self.apply_rotary_embedding(query_states, position_ids)
            key_states = self.apply_rotary_embedding(key_states, position_ids)

        # 处理KV缓存
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=1)
            value_states = torch.cat([past_value, value_states], dim=1)

        # GQA重复映射
        if self.num_key_value_heads != self.num_heads:
            key_states = torch.index_select(
                key_states, 2, self.kv_repeat_matrix.to(key_states.device)
            )
            value_states = torch.index_select(
                value_states, 2, self.kv_repeat_matrix.to(value_states.device)
            )

        # 调整维度顺序以进行注意力计算
        query_states = query_states.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        key_states = key_states.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        value_states = value_states.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]

        # 计算注意力分数
        attention_scores = torch.matmul(
            query_states, key_states.transpose(-1, -2)
        ) / math.sqrt(self.head_dim)

        # 应用注意力掩码
        if attention_mask is not None:
            # 确保掩码维度正确
            attention_scores = attention_scores + attention_mask

        # 注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 计算输出
        output = torch.matmul(attention_weights, value_states)  # [batch, num_heads, seq_len, head_dim]
        output = output.transpose(1, 2).contiguous()  # [batch, seq_len, num_heads, head_dim]
        output = output.view(batch_size, seq_length, -1)  # [batch, seq_len, d_model]
        output = self.o_proj(output)

        # 保存KV缓存
        current_key_value = (key_states, value_states) if past_key_value is not None else None

        return output, current_key_value


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, num_key_value_heads=None, mlp_ratio=4):
        super().__init__()
        self.attention = MultiHeadLatentAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_key_value_heads=num_key_value_heads
        )
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Linear(d_model * mlp_ratio, d_model)
        )
        self.attention_norm = nn.LayerNorm(d_model)
        self.mlp_norm = nn.LayerNorm(d_model)

    def forward(self, x, position_ids=None, attention_mask=None, past_key_value=None):
        # 注意力层
        residual = x
        x = self.attention_norm(x)
        x, current_key_value = self.attention(
            x,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_value=past_key_value
        )
        x = residual + x

        # MLP层
        residual = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = residual + x

        return x, current_key_value


class DeepSeekModel(nn.Module):
    def __init__(
            self,
            vocab_size,
            d_model,
            num_layers,
            num_heads,
            num_key_value_heads=None,
            mlp_ratio=4
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                num_key_value_heads=num_key_value_heads,
                mlp_ratio=mlp_ratio
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
            self,
            input_ids,
            position_ids=None,
            attention_mask=None,
            past_key_values=None
    ):
        # 获取输入嵌入
        x = self.embedding(input_ids)

        # 初始化past_key_values
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        current_key_values = []

        # 通过transformer层
        for i, layer in enumerate(self.layers):
            layer_past = past_key_values[i]
            x, current_key_value = layer(
                x,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_value=layer_past
            )
            current_key_values.append(current_key_value)

        # 最终输出
        x = self.norm(x)
        logits = self.lm_head(x)

        return logits, current_key_values


def test_model():
    # 模型参数
    vocab_size = 32000
    d_model = 512
    num_layers = 4
    num_heads = 8
    num_key_value_heads = 4
    batch_size = 2
    seq_length = 32

    # 创建模型
    model = DeepSeekModel(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads
    )

    # 创建输入
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    position_ids = torch.arange(seq_length).unsqueeze(0).expand(batch_size, -1)

    # 修改注意力掩码的创建方式
    attention_mask = torch.triu(
        torch.ones(seq_length, seq_length), diagonal=1
    ).bool()
    attention_mask = (
        attention_mask
        .unsqueeze(0)  # [1, seq_len, seq_len]
        .unsqueeze(0)  # [1, 1, seq_len, seq_len]
        .expand(batch_size, num_heads, -1, -1)  # [batch, num_heads, seq_len, seq_len]
        .to(dtype=torch.float32)
    )
    attention_mask = attention_mask * float('-inf')

    # 前向传播
    logits, key_value_cache = model(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=attention_mask
    )

    print("=== 模型测试结果 ===")
    print("输入形状:", input_ids.shape)
    print("输出形状:", logits.shape)
    print("缓存长度:", len(key_value_cache))

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print("\n总参数量:", total_params)

    # 测试GPU内存使用（如果可用）
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        model = model.cuda()
        input_ids = input_ids.cuda()
        position_ids = position_ids.cuda()
        attention_mask = attention_mask.cuda()

        start_mem = torch.cuda.memory_allocated()
        with torch.no_grad():
            logits, _ = model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask
            )
        end_mem = torch.cuda.memory_allocated()

        print("GPU内存使用:", (end_mem - start_mem) / 1024 / 1024, "MB")


if __name__ == "__main__":
    test_model()