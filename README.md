# ⚡ NanoReason
> Memory-Constrained LLM Inference & Reasoning Optimization Framework

NanoReason is a lightweight framework designed to enable **efficient reasoning using large language models on low-resource systems** (CPU-only, limited RAM environments).

---

## 🚀 Motivation

Modern LLMs are:
- Memory intensive (GB-scale models)
- Compute heavy (require GPUs)
- Inefficient for edge or constrained environments

NanoReason aims to solve this by:
- Reducing token usage
- Controlling reasoning flow
- Avoiding redundant computation

---

## 🧠 Core Idea

Instead of:
> "Use full model → process everything → waste compute"

NanoReason does:
> "Think selectively → compress input → reuse reasoning → optimize output"

---

## 🏗️ Architecture (v0.1)
Input Prompt
↓
Prompt Compressor (coming Day 2)
↓
Reasoning Controller (planned)
↓
Lightweight LLM (local)
↓
Memory Cache (planned)
↓
Output

---

## ⚙️ Tech Stack

- Python
- Hugging Face Transformers
- Lightweight models (e.g., distilgpt2)
- (Upcoming)
  - sentence-transformers
  - FAISS (vector memory)
  - llama.cpp / ctransformers

---

## 🧪 Day 1 Implementation

### ✔ Features Implemented
- Local LLM inference (CPU)
- Tokenization pipeline understanding
- Controlled generation using token limits

---

## 💻 Code Example

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

prompt = "Explain recursion in simple terms:"

inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=50)
outputs = model.generate(**inputs, max_length=100)

print(tokenizer.decode(outputs[0]))