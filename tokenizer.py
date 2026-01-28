class CharTokenizer:
    def __init__(self, text):
        # Create vocabulary from all unique characters in the text
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        
    def encode(self, s):
        # String to list of integers
        return [self.stoi[c] for c in s if c in self.stoi]
    
    def decode(self, l):
        # List of integers to string
        return ''.join([self.itos[i] for i in l])

if __name__ == "__main__":
    test_text = "<de> hallo welt! <en> hello world!"
    tokenizer = CharTokenizer(test_text)
    print(f"Vocab size: {tokenizer.vocab_size}")
    encoded = tokenizer.encode("<de> hallo")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {tokenizer.decode(encoded)}")
