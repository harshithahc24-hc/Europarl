import torch
import pickle
from model import MultilingualTransformer
from tokenizer import CharTokenizer

# Use same hyperparameters as train.py
n_embd = 128
n_head = 4
n_layer = 4
block_size = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_trained_model(model_path, tokenizer_path):
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    
    model = MultilingualTransformer(
        vocab_size=tokenizer.vocab_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        block_size=block_size
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_new_tokens=200):
    context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=max_new_tokens)[0].tolist()
    full_text = tokenizer.decode(generated)
    return full_text

if __name__ == "__main__":
    try:
        model, tokenizer = load_trained_model('model.pth', 'tokenizer.pkl')
        
        print("-" * 30)
        print("Generating English text...")
        print(generate_text(model, tokenizer, "<en> during the debate", 50))
        
        print("\n" + "-" * 30)
        print("Generating German text...")
        print(generate_text(model, tokenizer, "<de> während der debatte", 50))
        print("-" * 30)
        
    except FileNotFoundError:
        print("Model or Tokenizer file not found. Please run train.py first.")
