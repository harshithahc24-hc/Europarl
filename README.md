# Multilingual Text Generation from Scratch

This project implements a small Transformer-based language model for English and German text generation, trained on the Europarl corpus.

## Requirements
- Python 3.x
- PyTorch
- pickle (built-in)

## Project Structure
- `data_loader.py`: Handles loading and cleaning the Europarl dataset.
- `tokenizer.py`: A simple character-level tokenizer.
- `model.py`: Architecture of a decoder-only Transformer.
- `train.py`: Training logic and model saving.
- `generate.py`: Inference script to generate text in both languages.

## How to use
1. **Train the model**:
   Run `python train.py`. This will load a subset of the Europarl dataset, train for 2000 iterations (by default), and save `model.pth` and `tokenizer.pkl`.
   
2. **Generate text**:
   Run `python generate.py`. It will load the saved model and generate sample text for English and German using the respective tags `<en>` and `<de>`.

## Details
- **Architecture**: Decoder-only Transformer with Multi-Head Attention.
- **Tokenization**: Character-level (ensures small vocabulary and handles both languages easily).
- **Dataset**: Europarl v7 (English-German).
- **Multilingual Support**: Uses language tags to steer generation.
