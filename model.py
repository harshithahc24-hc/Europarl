import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters (can be moved to a config later)

class Head(nn.Module):
    """
    A single head of self-attention.
    
    Args:
        head_size (int): Dimensionality of the query, key, and value vectors.
        n_embd (int): Dimensionality of the input embeddings.
        block_size (int): Maximum sequence length (context window).
        dropout (float): Dropout probability for attention weights.
    """
    def __init__(self, head_size, n_embd, block_size, dropout=0.1):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass for a single attention head.
        
        Input shape: (Batch, Time, Channels)
        Output shape: (Batch, Time, Head_Size)
        """
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        
        # compute attention scores ("affinities")
        # scaled dot-product attention
        wei = q @ k.transpose(-2, -1) * (C**-0.5) 
        # apply causal mask (prevent looking into the future)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout=0.1):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class MultilingualTransformer(nn.Module):
    """
    A decoder-only Transformer model designed for multilingual text generation.
    It takes a sequence of token indices and predicts the next token in the sequence.
    
    Args:
        vocab_size (int): Size of the character vocabulary.
        n_embd (int): Embedding dimension.
        n_head (int): Number of attention heads.
        n_layer (int): Number of Transformer blocks.
        block_size (int): Maximum context window size.
        dropout (float): Dropout rate.
    """
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        """
        Forward pass for training or inference.
        
        Args:
            idx (Tensor): (B, T) tensor of token indices.
            targets (Tensor, optional): (B, T) tensor of target token indices for loss calculation.
            
        Returns:
            logits (Tensor): (B, T, Vocab_Size) raw prediction scores.
            loss (Tensor, optional): Cross-entropy loss if targets are provided.
        """
        B, T = idx.shape
        device = idx.device
        
        # tok_emb: (B, T, n_embd)
        tok_emb = self.token_embedding_table(idx) 
        # pos_emb: (T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) 
        
        x = tok_emb + pos_emb # (B, T, n_embd)
        x = self.blocks(x)    # (B, T, n_embd)
        x = self.ln_f(x)      # (B, T, n_embd)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        Generates new tokens following the provided context.
        
        Args:
            idx (Tensor): (B, T) tensor of starting token indices.
            max_new_tokens (int): Number of tokens to generate.
            
        Returns:
            Tensor: (B, T + max_new_tokens) indices of the full sequence.
        """
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
