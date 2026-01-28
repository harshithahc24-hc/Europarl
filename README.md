# 🌍 Europarl Multilingual Transformer

A high-performance, decoder-only Transformer model built from scratch using PyTorch. This project demonstrates bilingual text generation (English & German) using the [Europarl Corpus](https://www.statmt.org/europarl/), featuring character-level tokenization and language-tag steering.

---

## 🚀 Key Features

- **Custom Transformer Architecture**: Implements Multi-Head Attention, Positional Encodings, and Feed-Forward Networks from the ground up.
- **Bilingual Support**: Trained to recognize and generate text in both English (`<en>`) and German (`<de>`) using language-specific prefixes.
- **Character-Level Tokenization**: Ensures a compact vocabulary while maintaining the ability to represent any word in either language without "Out Of Vocabulary" errors.
- **Dynamic Training**: Optimized for both CPU and CUDA-enabled GPUs with automatic device detection.

---

## 🏗 Project Architecture

The model follows the modern "GPT-style" decoder-only architecture:

- **Embeddings**: Token + Positional embeddings (learned).
- **Transformer Blocks**: Multi-head self-attention with causal masking.
- **Normalization**: Pre-norm LayerNorm for improved stability.
- **Inference**: Temperature-controlled multinomial sampling for diverse text generation.

### File Structure:
- `model.py`: Core Transformer architecture (`Head`, `MultiHeadAttention`, `Block`, `MultilingualTransformer`).
- `data_loader.py`: Efficient streaming and cleaning of the parallel corpus.
- `tokenizer.py`: Character-to-integer mapping and decoding logic.
- `train.py`: Training pipeline with loss estimation and model checkpointing.
- `generate.py`: Inference script for real-time text generation.

---

## 🛠 Installation & Usage

### 1. Requirements
Ensure you have Python 3.8+ and PyTorch installed:
```bash
pip install torch
```

### 2. Dataset Preparation
The project expects the Europarl v7 files in an `archive/` directory:
- `archive/europarl-v7.de-en.de`
- `archive/europarl-v7.de-en.en`

### 3. Training the Model
Adjust hyperparameters in `train.py` (iterations, embedding size, layers) and run:
```bash
python train.py
```
*Current configuration: 4 layers, 128 embedding dimensions, 2000 iterations.*

### 4. Text Generation
Generate sample text using the pre-trained weights:
```bash
python generate.py
```

---

## 📊 Performance & Results

The current model (0.8M parameters) achieves a cross-entropy loss of **~1.53** after 2000 steps. 

### Sample Generations:
- **Prompt**: `<en> during the debate`
- **Output**: `during the debate is of the states the commission of the states regarding the report...`

- **Prompt**: `<de> während der debatte`
- **Output**: `während der debatte, die kommissiecher fïere, die ministerien der staaten zu...`

---

## 📜 License
This project is for educational purposes. The data is provided by the Europarl corpus project.
