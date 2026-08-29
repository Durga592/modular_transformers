```markdown
# Modular Transformers

A clean, production-ready, line-for-line modular PyTorch refactoring of [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/).

This repository breaks down the classic monolithic implementation into cleanly organized, highly readable modules—isolating attention mechanisms, model stacks, dataset wrappers, and training pipelines.

---

## 📁 Repository Structure

```text
modular-transformers/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── config/
│   └── default_config.yaml      # Model hyperparameters and training configuration
│
├── src/
│   ├── __init__.py
│   ├── modules/                 # Fine-grained neural network primitives
│   │   ├── __init__.py
│   │   ├── attention.py         # Scaled Dot-Product & Multi-Head Attention
│   │   ├── embeddings.py        # Token Embeddings & Positional Encodings
│   │   ├── feed_forward.py      # Position-wise Feed-Forward Network
│   │   └── layers.py            # LayerNorm & Sublayer Connections (Pre-LN)
│   │
│   ├── models/                  # Core Transformer architectures
│   │   ├── __init__.py
│   │   ├── encoder.py           # Encoder & EncoderLayer stacks
│   │   ├── decoder.py           # Decoder & DecoderLayer stacks
│   │   ├── transformer.py       # EncoderDecoder, Generator & make_model factory
│   │   └── utils.py             # Masking functions & architecture utilities
│   │
│   ├── data/                    # Dataset loaders and preprocessing
│   │   ├── __init__.py
│   │   ├── dataset.py           # Synthetic data generation & Batch processing
│   │   └── vocabulary.py        # Token mapping and vocab utilities
│   │
│   └── training/                # Training loop and loss functions
│       ├── __init__.py
│       ├── label_smoothing.py   # Label smoothing regularization
│       ├── lr_scheduler.py      # Custom Transformer warm-up scheduler
│       ├── trainer.py           # Epoch runner & state management
│       └── decode.py            # Greedy decoding inference loop
│
├── tests/                       # Unit tests for verification
│   ├── test_attention.py
│   ├── test_layers.py
│   └── test_model.py
│
└── train.py                     # Entry point script to launch training/eval

```

---

## 🚀 Getting Started with `uv`

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable virtual environment management and dependency installation.

### 1. Install `uv`

If you do not have `uv` installed, install it via the official installer or `pip`:

```bash
# macOS / Linux
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh

# Windows (PowerShell)
powershell -c "irm [https://astral.sh/uv/install.ps1](https://astral.sh/uv/install.ps1) | iex"

# Or via pip
pip install uv

```

### 2. Environment Setup & Dependency Installation

Clone the repository, create a virtual environment, and install all dependencies:

```bash
# Clone the repository
git clone [https://github.com/your-username/modular-transformers.git](https://github.com/your-username/modular-transformers.git)
cd modular-transformers

# Create a virtual environment using uv
uv venv

# Activate the virtual environment
# On macOS / Linux:
source .venv/bin/activate
# On Windows (PowerShell):
.venv\Scripts\activate

# Install dependencies fast using uv
uv pip install -r requirements.txt

# Download required spaCy language models
uv run python -m spacy download en_core_web_sm
uv run python -m spacy download de_core_news_sm

```

---

## 🧪 Testing

Verify that all architectural modules and forward passes function properly by running unit tests:

```bash
# Test Multi-Head Attention module
uv run python -m tests.test_attention

# Test LayerNorm and Sublayer connection primitives
uv run python -m tests.test_layers

# Test full EncoderDecoder Transformer instantiation
uv run python -m tests.test_model

```

---

## 🎯 Running Training & Inference

To execute the full synthetic copy-task training pipeline and run greedy decoding inference:

```bash
uv run train.py

```

---

## 🛠️ Architecture Highlights

* **Pre-LN Layer Normalization**: Applies `LayerNorm` before sub-layers rather than after, stabilizing deep stack optimization.
* **Scaled Dot-Product Attention**:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$


* **Multi-Head Attention**: Projects inputs into $h = 8$ parallel sub-spaces of dimension $d_k = 64$.
* **Positional Encoding**: Sinusoidal encodings added directly to word embeddings to preserve sequence order without recurrence.

---

## 📜 Acknowledgments

Based on the paper *"Attention Is All You Need"* (Vaswani et al., 2017) and implementation references from [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/).
