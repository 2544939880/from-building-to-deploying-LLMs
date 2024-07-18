from importlib.metadata import version
import torch
import torch.nn as nn
import tiktoken
import re
from einops import rearrange
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, sample_length, stride) -> None:
        super().__init__()
        self.input_idx = []
        self.target_idx = []

        token_idx = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        for i in  range(0, len(token_idx) - sample_length + 1, stride):
            input_chunk = token_idx[i: i + sample_length]
            target_chunk = token_idx[i + 1: i + sample_length + 1]
            self.input_idx.append(torch.tensor(input_chunk))
            self.target_idx.append(torch.tensor(target_chunk))
    def __len__(self):
        return len(self.target_idx)
    
    def __getitem__(self, index):
        return self.input_idx[index], self.target_idx[index]
    
def create_dataloader_V1(txt, batch_size=4, sample_length=256, stride=128, 
                         shuffle=True, drop_last=True, num_workers = 7):
    
    tokenizer = tiktoken.get_encoding('gpt2')
    
    dataset = GPTDatasetV1(txt, tokenizer, sample_length, stride)

    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=shuffle, 
                            drop_last=drop_last, 
                            num_workers=num_workers, 
                            persistent_workers=True)
    
    return dataloader

class SelfAttention(nn.Module):
    # d_in: input token embed dim, d_out: q, k, and v space dim
    def __init__(self, d_in, d_out, qkv_bias=False) -> None:
        super().__init__()
        self.d_k = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
    
    def forward(self, x):
        # x shape: [batch_size, sample_length, d_in]
        # batch_size, sample_length, d_in = x.shape()

        # q, k, and v shape: [batch_size, sample_length, d_out]
        queries = self.W_query(x) 
        keys = self.W_key(x)
        values = self.W_value(x)

        # attn_scores shape: [batch_size, sample_length, sample_length]
        attn_scores = queries @ keys.transpose(1, 2)
        attn_weights = torch.softmax(attn_scores * (self.d_k ** -0.5), dim=-1)
        attention = attn_weights @ values

        return attention

class CausalAttention(nn.Module):
    # d_in: input token embed dim, d_out: q, k, and v space dim
    def __init__(self, d_in, d_out, sample_length, dropout, qkv_bias = False) -> None:
        super().__init__()
        self.d_k = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', 
                             torch.triu(torch.ones(sample_length, sample_length), diagonal=1))
    
    def forward(self, x):
        # x shape: [batch_size, sample_length, d_in]
        # batch_size, sample_length, d_in = x.shape()

        queries = self.W_query(x)
        keys  = self.W_key(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(self.mask.bool(), -torch.inf)
        attn_weights = torch.softmax(attn_scores * (self.d_k ** -0.5), dim=-1)

        attn_weights = self.dropout(attn_weights)
        attention = attn_weights @ values

        return attention
    
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, num_heads, qkv_bias=False):
      super().__init__()
      self.heads = nn.ModuleList(
          [SelfAttention(d_in, d_out, qkv_bias) for _ in range(num_heads)]
      )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)

class MultiHeadCausalAttention(nn.Module):
    # d_in: input token embed dim, d_out: q, k, and v space dim
    def __init__(self, d_in, d_out, sample_length, num_heads, dropout, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = d_out / num_heads
        self.d_k = d_out
        
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', 
                             torch.triu(torch.ones(sample_length, sample_length), diagonal=1))
        self.out_proj = nn.Linear(d_out, d_out)

    def forward(self, x):
        # x shape: [batch_size, sample_length, d_in]
        # batch_size, sample_length, d_in = x.shape()

        # q, k, and v shape: [batch_size, sample_length, d_out]
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # 我们通过添加“num_heads”维度来隐式地分割矩阵
        # q, k, and v shape: [batch_size, num_heads, sample_length, head_dim]
        queries, keys, values = map(
            lambda x: rearrange(x, 'b s (h d) -> b h s d', h=self.num_heads),
            [queries, keys, values]
        )

        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores.masked_fill_(self.mask.bool(), -torch.inf)
        attn_weights = torch.softmax(attn_scores * (self.d_k ** -0.5), dim=-1)

        attn_weights = self.dropout(attn_weights)
        attention = attn_weights @ values

        # attention shape: [batch_size, sample_length, d_out]
        attention = rearrange(attention, 'b h s d -> b s (h d)')
        attention = self.out_proj(attention)

        return attention

if __name__ == '__main__':
    # Print package version
    print("Pytorch version: ", torch.__version__)
    print("Tiktoken version: ", version("tiktoken"))

    # read the raw text 
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    # item.strip() 去除每个拆分项（item）的前后空白字符。
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    # print(preprocessed[:30], len(preprocessed))

    # GPT-2 used BytePair encoding (BPE) as its tokenizer
    # 它允许模型将预定义词汇表中不存在的单词分解为更小的子词单元甚至单个字符
    # 从而使其能够处理词汇表之外的单词
    tokenizer = tiktoken.get_encoding('gpt2')

    text = (
        "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
        "of someunknownPlace."
    )

    integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    # print(integers, type(integers))

    strings = [tokenizer.decode([item]) for item in integers]
    # print(strings)

    sample_length = 1024

    dataloader = create_dataloader_V1(raw_text, 
                                      sample_length=sample_length, 
                                      stride=1)

    vocal_size = tokenizer.n_vocab
    embedding_dim = 256

    token_embedding_layer = torch.nn.Embedding(vocal_size, embedding_dim)

    # Absolute position embeddings
    position_embedding_layer = torch.nn.Embedding(sample_length, embedding_dim)
    pos_embeddings =  position_embedding_layer(torch.arange(sample_length))

    for i, (input, target) in enumerate(dataloader):

        # input and target shape: [batch_size, sample_length]
        token_embeddings = token_embedding_layer(input)
        print("Input batch size shape: ", input.shape)
        
        # LLM模型输入嵌入张量 = Token embeddings + Position emdeddings
        # input_embeddings shape: [batch_size, sample_length, embedding_dim]
        input_embeddings = token_embeddings + pos_embeddings
        print("Embed input shape:", input_embeddings.shape)
        
        attention = MultiHeadCausalAttention(
            d_in=embedding_dim, 
            d_out=512, 
            sample_length=sample_length, 
            num_heads=4, 
            dropout=0.0
        )

        output = attention(input_embeddings)
        print("Attention mapping shape: ", output.shape)
        break
