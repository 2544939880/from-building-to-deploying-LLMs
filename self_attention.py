import torch
import torch.utils

# Step1: 初始化输入张量维度
sample_length = 256
embedding_dim = 128
batch_size = 16

input = torch.rand(batch_size, sample_length, embedding_dim)

# Step2: 设置Q，K，V采样矩阵
d_in = embedding_dim
d_out = 64

W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

# Step3: 单位样本注意力计算演示
sample = input[1]

querys = sample @ W_query
keys = sample @ W_key
values = sample @ W_value

d_k = keys.shape[1]

attn_scores = querys @ keys.T
attn_weights = torch.softmax(attn_scores * (d_k ** -0.5), dim=-1)
attention = attn_weights @ values
print("Attention shape: ", attention.shape)
