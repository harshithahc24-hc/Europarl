import os

def load_data(de_path, en_path, num_samples=10000):
    """
    Loads parallel English-German text, adds language tags, and returns a combined list.
    """
    print(f"Loading {num_samples} samples from {de_path} and {en_path}...")
    
    from itertools import islice
    with open(de_path, 'r', encoding='utf-8') as f_de, \
         open(en_path, 'r', encoding='utf-8') as f_en:
        
        de_lines = [line.strip() for line in islice(f_de, num_samples)]
        en_lines = [line.strip() for line in islice(f_en, num_samples)]
        
    # Combine with language tags
    # We want the model to learn to generate in both languages
    # One approach is to just have a corpus of tagged sentences.
    # We can also pair them, but for a general language model, 
    # we just need a sequence of text with markers.
    
    data = []
    for de, en in zip(de_lines, en_lines):
        if de and en:
            data.append(f"<de> {de}")
            data.append(f"<en> {en}")
            
    return data

def clean_text(text_list):
    """
    Simple cleaning: lowercasing and basic character filtering.
    """
    cleaned = []
    for text in text_list:
        # Keep it simple for character-level models
        cleaned.append(text.lower())
    return cleaned

if __name__ == "__main__":
    # Test loading a small subset
    base_path = r"c:\Users\manoj\OneDrive\Desktop\Europarl languages\archive"
    de_file = os.path.join(base_path, "europarl-v7.de-en.de")
    en_file = os.path.join(base_path, "europarl-v7.de-en.en")
    
    samples = load_data(de_file, en_file, num_samples=5)
    cleaned = clean_text(samples)
    for s in cleaned:
        print(s)
