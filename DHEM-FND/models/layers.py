import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import math


class ReverseLayerF(Function):
    """梯度反转层（用于领域对抗训练）"""

    @staticmethod
    def forward(ctx, input, alpha):
        ctx.save_for_backward(input, alpha)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input, alpha = ctx.saved_tensors
        return grad_output.neg() * alpha, None


class MLP(nn.Module):
    """多层感知机"""

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = []
        current_dim = input_dim

        # 构建隐藏层
        for dim in embed_dims:
            layers.extend([
                nn.Linear(current_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            current_dim = dim

        # 添加输出层
        if output_layer:
            layers.append(nn.Linear(current_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class cnn_extractor(nn.Module):
    """CNN特征提取器"""

    def __init__(self, feature_kernel, input_size):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(input_size, num, kernel)
            for kernel, num in feature_kernel.items()
        ])

    def forward(self, x):
        # 调整维度: (batch, channels, seq_len)
        x = x.permute(0, 2, 1)

        # 提取多尺度特征
        features = [F.relu(conv(x)) for conv in self.convs]
        features = [F.max_pool1d(f, f.size(-1)).squeeze(-1) for f in features]

        return torch.cat(features, dim=1)


class MaskAttention(nn.Module):
    """带掩码的注意力机制"""

    def __init__(self, input_size):
        super().__init__()
        self.attn = nn.Linear(input_size, 1)

    def forward(self, inputs, mask=None):
        # 计算注意力分数 [batch, seq_len]
        scores = self.attn(inputs).squeeze(-1)

        # 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 计算注意力权重 [batch, 1, seq_len]
        attn_weights = F.softmax(scores, dim=-1).unsqueeze(1)

        # 上下文向量 [batch, hidden_size]
        context = torch.bmm(attn_weights, inputs).squeeze(1)

        return context, attn_weights


class Attention(nn.Module):
    """缩放点积注意力"""

    def forward(self, query, key, value, mask=None, dropout=None):
        # 计算注意力分数
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))

        # 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 计算注意力权重
        attn_weights = F.softmax(scores, dim=-1)

        # 应用dropout
        if dropout is not None:
            attn_weights = dropout(attn_weights)

        return torch.matmul(attn_weights, value), attn_weights


class MultiHeadedAttention(nn.Module):
    """多头注意力机制"""

    def __init__(self, num_heads, d_model, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"

        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        # 线性变换层
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 线性投影并分头
        query, key, value = [
            lin(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 应用注意力
        x, attn = Attention()(
            query, key, value,
            mask=mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1) if mask else None,
            dropout=self.dropout
        )

        # 合并多头输出
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.output_linear(x), attn


class SelfAttentionFeatureExtract(nn.Module):
    """自注意力特征提取器"""

    def __init__(self, num_heads, input_size, output_size):
        super().__init__()
        self.attention = MultiHeadedAttention(num_heads, input_size)
        self.out_layer = nn.Linear(input_size, output_size)

    def forward(self, inputs, query, mask=None):
        # 应用多头注意力
        features, attn = self.attention(
            query=query,
            key=inputs,
            value=inputs,
            mask=mask.unsqueeze(1).unsqueeze(2) if mask is not None else None
        )

        # 输出变换
        return self.out_layer(features), attn