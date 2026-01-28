# 🛠 Project Tools & Technologies

This document outlines the software, libraries, and architectural components used to build and train the Europarl Multilingual Transformer.

---

## 💻 Core Technologies
- **Python 3.13**: The primary programming language used for logic, data processing, and model implementation.
- **PyTorch**: The deep learning framework used to build the Transformer architecture, manage tensor operations, and handle GPU/CPU training.
- **Git**: Used for version control, allowing for tracking changes and managing the project's evolution.
- **GitHub**: The hosting platform for the remote repository, used for collaboration and code storage.

## 🏗 Model Architecture (Transformer)
- **Decoder-Only Transformer**: A generative architecture (similar to GPT) that predicts the next character based on previous context.
- **Multi-Head Self-Attention**: Allows the model to focus on different parts of a sentence simultaneously to understand context.
- **Causal Masking**: Ensures the model only looks at "past" characters during training, preventing it from "cheating" by seeing the future.
- **Positional Encoding**: Learnable embeddings that tell the model where each character is located in a sequence.
- **Layer Normalization**: Used to stabilize the training process and improve convergence.

## 📊 Data & Tokenization
- **Europarl v7 Corpus**: A high-quality parallel dataset of parliamentary proceedings from the European Union (English and German).
- **Character-Level Tokenizer**: A custom-built tokenizer that maps individual characters to integers, allowing the model to handle any word without a fixed dictionary.
- **Language Tagging**: Special tokens (`<en>`, `<de>`) used to "steer" the model to generate text in a specific language.

## 🚂 Training & Optimization
- **AdamW Optimizer**: An advanced optimization algorithm with weight decay to prevent overfitting.
- **Softmax Cross-Entropy Loss**: The mathematical function used to measure how well the model's predictions match the actual text.
- **Pickle**: A Python module used to serialize and save the `CharTokenizer` object so it can be reused during generation.

## 🛠 Development Workflow Tools
- **PowerShell/Terminal**: Used for executing scripts and managing Git commands.
- **Visual Studio Code (VS Code)**: The Integrated Development Environment (IDE) used for writing and debugging the code.
- **Antigravity (AI Assistant)**: Used for automated code generation, bug fixing, and repository management.
