import torch
import torch.optim as optim
from data_loader import load_data, clean_text
from tokenizer import CharTokenizer
from model import MultilingualTransformer
import os

# Hyperparameters
batch_size = 16
block_size = 64
max_iters = 2000
eval_interval = 200
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 20
n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.2
num_samples = 1000 # Number of lines from each file

def get_batch(data, batch_size, block_size):
    # generate a small batch of data of inputs x and targets y
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, block_size):
    out = {}
    model.eval()
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, batch_size, block_size)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def train():
    # 1. Load and clean data
    base_path = r"c:\Users\manoj\OneDrive\Desktop\Europarl languages\archive"
    de_file = os.path.join(base_path, "europarl-v7.de-en.de")
    en_file = os.path.join(base_path, "europarl-v7.de-en.en")
    
    raw_data = load_data(de_file, en_file, num_samples=num_samples)
    cleaned_data = clean_text(raw_data)
    
    # Combine everything into one giant string for character-level model
    full_text = "\n".join(cleaned_data)
    
    # 2. Tokenization
    tokenizer = CharTokenizer(full_text)
    data_tensor = torch.tensor(tokenizer.encode(full_text), dtype=torch.long)
    
    # Split into train and validation
    n = int(0.9 * len(data_tensor))
    train_data = data_tensor[:n]
    val_data = data_tensor[n:]
    
    print(f"Data size: {len(data_tensor)} characters")
    print(f"Vocab size: {tokenizer.vocab_size}")
    
    # 3. Model architecture
    model = MultilingualTransformer(
        vocab_size=tokenizer.vocab_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        block_size=block_size,
        dropout=dropout
    )
    model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # 4. Training loop
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    for iter in range(max_iters):
        # Every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data, batch_size, block_size)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch(train_data, batch_size, block_size)

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Save the model and tokenizer info
    torch.save(model.state_dict(), 'model.pth')
    # Save tokenizer vocab
    import pickle
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
        
    print("Training complete. Model saved to model.pth")

if __name__ == "__main__":
    train()
