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

## 📊 Training Results & Experiments

I evaluated the model across **10, 20, and 30 epochs** on the synthetic copy task to track loss convergence and decoding accuracy.



### 📉 Loss Curves and 📈 Epoch Comparisons

Below is the comparison of training and validation loss over 10, 20, and 30 epochs:

> **Observations:**
> - **10 Epochs:** Initial convergence phase; model learns basic sequence tokens but still exhibits minor loss variance.

```
Train worker process using device: mps
[mps] Epoch 0 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   8.10 | Tokens / Sec:   254.0 | Learning Rate: 5.4e-07
Epoch Step:     41 | Accumulation Step:   5 | Loss:   7.87 | Tokens / Sec:   559.1 | Learning Rate: 1.1e-05
Epoch Step:     81 | Accumulation Step:   9 | Loss:   7.54 | Tokens / Sec:   385.3 | Learning Rate: 2.2e-05
Epoch Step:    121 | Accumulation Step:  13 | Loss:   7.37 | Tokens / Sec:   378.9 | Learning Rate: 3.3e-05
Epoch Step:    161 | Accumulation Step:  17 | Loss:   7.11 | Tokens / Sec:   384.4 | Learning Rate: 4.4e-05
Epoch Step:    201 | Accumulation Step:  21 | Loss:   6.91 | Tokens / Sec:   390.5 | Learning Rate: 5.4e-05
Epoch Step:    241 | Accumulation Step:  25 | Loss:   6.77 | Tokens / Sec:   380.0 | Learning Rate: 6.5e-05
Epoch Step:    281 | Accumulation Step:  29 | Loss:   6.54 | Tokens / Sec:   382.5 | Learning Rate: 7.6e-05
Epoch Step:    321 | Accumulation Step:  33 | Loss:   6.51 | Tokens / Sec:   381.5 | Learning Rate: 8.7e-05
Epoch Step:    361 | Accumulation Step:  37 | Loss:   6.17 | Tokens / Sec:   386.2 | Learning Rate: 9.7e-05
Epoch Step:    401 | Accumulation Step:  41 | Loss:   5.94 | Tokens / Sec:   390.4 | Learning Rate: 1.1e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   5.64 | Tokens / Sec:   382.7 | Learning Rate: 1.2e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   5.48 | Tokens / Sec:   382.8 | Learning Rate: 1.3e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   5.24 | Tokens / Sec:   392.2 | Learning Rate: 1.4e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   4.92 | Tokens / Sec:   389.9 | Learning Rate: 1.5e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   4.54 | Tokens / Sec:   394.8 | Learning Rate: 1.6e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   4.76 | Tokens / Sec:   389.2 | Learning Rate: 1.7e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   4.72 | Tokens / Sec:   378.7 | Learning Rate: 1.8e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   4.46 | Tokens / Sec:   383.5 | Learning Rate: 1.9e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   4.47 | Tokens / Sec:   381.5 | Learning Rate: 2.0e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   4.33 | Tokens / Sec:   380.3 | Learning Rate: 2.2e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   4.31 | Tokens / Sec:   393.4 | Learning Rate: 2.3e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   4.18 | Tokens / Sec:   387.6 | Learning Rate: 2.4e-04
[mps] Epoch 0 Validation ====
(tensor(4.1534, device='mps:0'), <__main__.TrainState object at 0x1257391c0>)
[mps] Epoch 1 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   4.30 | Tokens / Sec:   423.1 | Learning Rate: 2.4e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   4.21 | Tokens / Sec:   426.1 | Learning Rate: 2.6e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   4.20 | Tokens / Sec:   393.0 | Learning Rate: 2.7e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   3.79 | Tokens / Sec:   388.5 | Learning Rate: 2.8e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   3.80 | Tokens / Sec:   389.0 | Learning Rate: 2.9e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   3.78 | Tokens / Sec:   393.6 | Learning Rate: 3.0e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   3.93 | Tokens / Sec:   386.5 | Learning Rate: 3.1e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   3.95 | Tokens / Sec:   386.0 | Learning Rate: 3.2e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   4.04 | Tokens / Sec:   382.5 | Learning Rate: 3.3e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   3.91 | Tokens / Sec:   381.1 | Learning Rate: 3.4e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   3.72 | Tokens / Sec:   389.2 | Learning Rate: 3.5e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   3.65 | Tokens / Sec:   377.1 | Learning Rate: 3.6e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   3.56 | Tokens / Sec:   385.0 | Learning Rate: 3.7e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   3.35 | Tokens / Sec:   377.7 | Learning Rate: 3.8e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   3.45 | Tokens / Sec:   389.0 | Learning Rate: 4.0e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   3.42 | Tokens / Sec:   378.9 | Learning Rate: 4.1e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   3.49 | Tokens / Sec:   388.5 | Learning Rate: 4.2e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   3.58 | Tokens / Sec:   386.1 | Learning Rate: 4.3e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   3.32 | Tokens / Sec:   386.2 | Learning Rate: 4.4e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   3.27 | Tokens / Sec:   387.0 | Learning Rate: 4.5e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   3.37 | Tokens / Sec:   388.3 | Learning Rate: 4.6e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   3.11 | Tokens / Sec:   390.5 | Learning Rate: 4.7e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   3.20 | Tokens / Sec:   387.4 | Learning Rate: 4.8e-04
[mps] Epoch 1 Validation ====
(tensor(3.1085, device='mps:0'), <__main__.TrainState object at 0x1257391c0>)
[mps] Epoch 2 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   3.01 | Tokens / Sec:   403.2 | Learning Rate: 4.9e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   2.89 | Tokens / Sec:   408.2 | Learning Rate: 5.0e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   2.89 | Tokens / Sec:   385.6 | Learning Rate: 5.1e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   2.91 | Tokens / Sec:   382.4 | Learning Rate: 5.2e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   3.11 | Tokens / Sec:   389.0 | Learning Rate: 5.3e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   2.95 | Tokens / Sec:   386.4 | Learning Rate: 5.4e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   2.99 | Tokens / Sec:   379.9 | Learning Rate: 5.5e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   2.85 | Tokens / Sec:   389.8 | Learning Rate: 5.6e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   2.64 | Tokens / Sec:   386.9 | Learning Rate: 5.7e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   2.63 | Tokens / Sec:   381.9 | Learning Rate: 5.9e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   2.46 | Tokens / Sec:   383.0 | Learning Rate: 6.0e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   2.79 | Tokens / Sec:   385.8 | Learning Rate: 6.1e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   2.83 | Tokens / Sec:   388.3 | Learning Rate: 6.2e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   2.60 | Tokens / Sec:   388.2 | Learning Rate: 6.3e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   2.60 | Tokens / Sec:   383.7 | Learning Rate: 6.4e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   2.90 | Tokens / Sec:   387.3 | Learning Rate: 6.5e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   2.27 | Tokens / Sec:   380.7 | Learning Rate: 6.6e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   2.62 | Tokens / Sec:   384.6 | Learning Rate: 6.7e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   2.75 | Tokens / Sec:   377.6 | Learning Rate: 6.8e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   2.41 | Tokens / Sec:   388.7 | Learning Rate: 6.9e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   2.63 | Tokens / Sec:   389.6 | Learning Rate: 7.0e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   2.73 | Tokens / Sec:   382.3 | Learning Rate: 7.1e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   2.38 | Tokens / Sec:   376.5 | Learning Rate: 7.3e-04
[mps] Epoch 2 Validation ====
(tensor(2.3598, device='mps:0'), <__main__.TrainState object at 0x1257391c0>)
[mps] Epoch 3 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   2.05 | Tokens / Sec:   409.8 | Learning Rate: 7.3e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   2.11 | Tokens / Sec:   413.1 | Learning Rate: 7.4e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   2.01 | Tokens / Sec:   384.9 | Learning Rate: 7.5e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   2.37 | Tokens / Sec:   384.5 | Learning Rate: 7.6e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   2.27 | Tokens / Sec:   379.9 | Learning Rate: 7.8e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   2.45 | Tokens / Sec:   385.9 | Learning Rate: 7.9e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   1.87 | Tokens / Sec:   397.0 | Learning Rate: 8.0e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   2.25 | Tokens / Sec:   381.9 | Learning Rate: 8.1e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   2.09 | Tokens / Sec:   390.2 | Learning Rate: 8.0e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   2.25 | Tokens / Sec:   375.3 | Learning Rate: 8.0e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   1.87 | Tokens / Sec:   379.9 | Learning Rate: 7.9e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   2.12 | Tokens / Sec:   383.3 | Learning Rate: 7.9e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   2.02 | Tokens / Sec:   389.9 | Learning Rate: 7.8e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   1.98 | Tokens / Sec:   386.4 | Learning Rate: 7.8e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   2.27 | Tokens / Sec:   384.6 | Learning Rate: 7.7e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   1.97 | Tokens / Sec:   385.3 | Learning Rate: 7.7e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   2.22 | Tokens / Sec:   386.6 | Learning Rate: 7.6e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   2.00 | Tokens / Sec:   388.2 | Learning Rate: 7.6e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   2.05 | Tokens / Sec:   383.8 | Learning Rate: 7.5e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   1.80 | Tokens / Sec:   382.8 | Learning Rate: 7.5e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   1.86 | Tokens / Sec:   388.9 | Learning Rate: 7.4e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   2.02 | Tokens / Sec:   380.8 | Learning Rate: 7.4e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   2.13 | Tokens / Sec:   388.5 | Learning Rate: 7.4e-04
[mps] Epoch 3 Validation ====
(tensor(1.9325, device='mps:0'), <__main__.TrainState object at 0x1257391c0>)
[mps] Epoch 4 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   1.81 | Tokens / Sec:   526.9 | Learning Rate: 7.3e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   1.57 | Tokens / Sec:   425.6 | Learning Rate: 7.3e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   1.69 | Tokens / Sec:   398.9 | Learning Rate: 7.3e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   1.69 | Tokens / Sec:   384.0 | Learning Rate: 7.2e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   1.50 | Tokens / Sec:   385.9 | Learning Rate: 7.2e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   1.95 | Tokens / Sec:   386.1 | Learning Rate: 7.1e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   1.59 | Tokens / Sec:   387.7 | Learning Rate: 7.1e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   1.62 | Tokens / Sec:   387.0 | Learning Rate: 7.1e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   1.83 | Tokens / Sec:   391.3 | Learning Rate: 7.0e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   1.45 | Tokens / Sec:   374.9 | Learning Rate: 7.0e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   1.74 | Tokens / Sec:   385.0 | Learning Rate: 7.0e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   1.80 | Tokens / Sec:   378.9 | Learning Rate: 6.9e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   1.71 | Tokens / Sec:   381.3 | Learning Rate: 6.9e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   1.76 | Tokens / Sec:   392.1 | Learning Rate: 6.9e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   1.76 | Tokens / Sec:   385.6 | Learning Rate: 6.8e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   1.76 | Tokens / Sec:   390.2 | Learning Rate: 6.8e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   1.54 | Tokens / Sec:   390.8 | Learning Rate: 6.8e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   1.37 | Tokens / Sec:   387.3 | Learning Rate: 6.7e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   1.77 | Tokens / Sec:   385.4 | Learning Rate: 6.7e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   1.50 | Tokens / Sec:   400.5 | Learning Rate: 6.7e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   1.63 | Tokens / Sec:   388.6 | Learning Rate: 6.6e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   1.32 | Tokens / Sec:   387.0 | Learning Rate: 6.6e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   1.42 | Tokens / Sec:   396.7 | Learning Rate: 6.6e-04
[mps] Epoch 4 Validation ====
(tensor(1.7533, device='mps:0'), <__main__.TrainState object at 0x1257391c0>)
[mps] Epoch 5 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   1.34 | Tokens / Sec:   478.0 | Learning Rate: 6.6e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   1.18 | Tokens / Sec:   415.1 | Learning Rate: 6.5e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   1.23 | Tokens / Sec:   387.4 | Learning Rate: 6.5e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   1.34 | Tokens / Sec:   395.7 | Learning Rate: 6.5e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   1.48 | Tokens / Sec:   399.4 | Learning Rate: 6.4e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   1.48 | Tokens / Sec:   380.6 | Learning Rate: 6.4e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   1.30 | Tokens / Sec:   386.3 | Learning Rate: 6.4e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   1.37 | Tokens / Sec:   387.3 | Learning Rate: 6.4e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   1.46 | Tokens / Sec:   380.2 | Learning Rate: 6.3e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   1.52 | Tokens / Sec:   388.7 | Learning Rate: 6.3e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   1.46 | Tokens / Sec:   395.4 | Learning Rate: 6.3e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   1.24 | Tokens / Sec:   386.4 | Learning Rate: 6.3e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   1.21 | Tokens / Sec:   396.1 | Learning Rate: 6.2e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   1.31 | Tokens / Sec:   386.6 | Learning Rate: 6.2e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   1.30 | Tokens / Sec:   393.5 | Learning Rate: 6.2e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   1.57 | Tokens / Sec:   389.1 | Learning Rate: 6.2e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   1.41 | Tokens / Sec:   392.4 | Learning Rate: 6.1e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   1.18 | Tokens / Sec:   392.7 | Learning Rate: 6.1e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   1.26 | Tokens / Sec:   396.9 | Learning Rate: 6.1e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   1.20 | Tokens / Sec:   387.5 | Learning Rate: 6.1e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   1.37 | Tokens / Sec:   386.2 | Learning Rate: 6.0e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   1.25 | Tokens / Sec:   391.5 | Learning Rate: 6.0e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   1.42 | Tokens / Sec:   382.8 | Learning Rate: 6.0e-04
[mps] Epoch 5 Validation ====
(tensor(1.6735, device='mps:0'), <__main__.TrainState object at 0x1257391c0>)
[mps] Epoch 6 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   1.08 | Tokens / Sec:   460.8 | Learning Rate: 6.0e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   1.15 | Tokens / Sec:   398.4 | Learning Rate: 6.0e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   1.16 | Tokens / Sec:   406.3 | Learning Rate: 5.9e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   1.17 | Tokens / Sec:   388.9 | Learning Rate: 5.9e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   1.13 | Tokens / Sec:   386.5 | Learning Rate: 5.9e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   1.02 | Tokens / Sec:   389.7 | Learning Rate: 5.9e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   1.10 | Tokens / Sec:   386.8 | Learning Rate: 5.9e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   1.13 | Tokens / Sec:   384.6 | Learning Rate: 5.8e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   1.07 | Tokens / Sec:   379.0 | Learning Rate: 5.8e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   1.23 | Tokens / Sec:   390.6 | Learning Rate: 5.8e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   1.07 | Tokens / Sec:   382.8 | Learning Rate: 5.8e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   1.16 | Tokens / Sec:   386.4 | Learning Rate: 5.8e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   1.29 | Tokens / Sec:   385.4 | Learning Rate: 5.7e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   1.21 | Tokens / Sec:   380.9 | Learning Rate: 5.7e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   1.31 | Tokens / Sec:   379.7 | Learning Rate: 5.7e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   1.12 | Tokens / Sec:   383.8 | Learning Rate: 5.7e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   1.19 | Tokens / Sec:   380.4 | Learning Rate: 5.7e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   1.03 | Tokens / Sec:   387.3 | Learning Rate: 5.6e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   1.31 | Tokens / Sec:   378.2 | Learning Rate: 5.6e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   1.07 | Tokens / Sec:   376.0 | Learning Rate: 5.6e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   1.05 | Tokens / Sec:   384.3 | Learning Rate: 5.6e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   1.40 | Tokens / Sec:   391.7 | Learning Rate: 5.6e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   1.14 | Tokens / Sec:   376.3 | Learning Rate: 5.6e-04
[mps] Epoch 6 Validation ====
(tensor(1.6380, device='mps:0'), <__main__.TrainState object at 0x1257391c0>)
[mps] Epoch 7 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   1.05 | Tokens / Sec:   476.6 | Learning Rate: 5.5e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   0.96 | Tokens / Sec:   405.2 | Learning Rate: 5.5e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   0.90 | Tokens / Sec:   400.4 | Learning Rate: 5.5e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   0.97 | Tokens / Sec:   394.5 | Learning Rate: 5.5e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   0.84 | Tokens / Sec:   377.4 | Learning Rate: 5.5e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   0.88 | Tokens / Sec:   390.6 | Learning Rate: 5.5e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   1.09 | Tokens / Sec:   392.5 | Learning Rate: 5.4e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   1.04 | Tokens / Sec:   382.6 | Learning Rate: 5.4e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   0.93 | Tokens / Sec:   385.0 | Learning Rate: 5.4e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   1.03 | Tokens / Sec:   389.8 | Learning Rate: 5.4e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   1.08 | Tokens / Sec:   384.5 | Learning Rate: 5.4e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   1.06 | Tokens / Sec:   392.1 | Learning Rate: 5.4e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   0.85 | Tokens / Sec:   380.6 | Learning Rate: 5.3e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   1.00 | Tokens / Sec:   392.0 | Learning Rate: 5.3e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   0.96 | Tokens / Sec:   385.6 | Learning Rate: 5.3e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   0.92 | Tokens / Sec:   392.1 | Learning Rate: 5.3e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   0.89 | Tokens / Sec:   389.6 | Learning Rate: 5.3e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   1.03 | Tokens / Sec:   394.1 | Learning Rate: 5.3e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   1.03 | Tokens / Sec:   392.3 | Learning Rate: 5.3e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   0.82 | Tokens / Sec:   380.5 | Learning Rate: 5.2e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   0.90 | Tokens / Sec:   384.7 | Learning Rate: 5.2e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   1.23 | Tokens / Sec:   391.9 | Learning Rate: 5.2e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   0.97 | Tokens / Sec:   390.1 | Learning Rate: 5.2e-04
[mps] Epoch 7 Validation ====
(tensor(1.6429, device='mps:0'), <__main__.TrainState object at 0x1257391c0>)
[mps] Epoch 8 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   0.86 | Tokens / Sec:   441.9 | Learning Rate: 5.2e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   0.90 | Tokens / Sec:   405.5 | Learning Rate: 5.2e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   0.95 | Tokens / Sec:   402.2 | Learning Rate: 5.2e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   0.99 | Tokens / Sec:   384.4 | Learning Rate: 5.1e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   0.87 | Tokens / Sec:   384.3 | Learning Rate: 5.1e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   0.88 | Tokens / Sec:   384.6 | Learning Rate: 5.1e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   0.73 | Tokens / Sec:   386.7 | Learning Rate: 5.1e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   0.85 | Tokens / Sec:   380.7 | Learning Rate: 5.1e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   0.76 | Tokens / Sec:   383.1 | Learning Rate: 5.1e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   0.76 | Tokens / Sec:   376.5 | Learning Rate: 5.1e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   0.92 | Tokens / Sec:   382.0 | Learning Rate: 5.1e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   0.89 | Tokens / Sec:   388.0 | Learning Rate: 5.0e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   0.84 | Tokens / Sec:   386.1 | Learning Rate: 5.0e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   0.81 | Tokens / Sec:   383.9 | Learning Rate: 5.0e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   0.88 | Tokens / Sec:   376.0 | Learning Rate: 5.0e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   0.98 | Tokens / Sec:   388.9 | Learning Rate: 5.0e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   1.06 | Tokens / Sec:   380.8 | Learning Rate: 5.0e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   0.80 | Tokens / Sec:   382.1 | Learning Rate: 5.0e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   0.93 | Tokens / Sec:   389.4 | Learning Rate: 4.9e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   0.96 | Tokens / Sec:   379.8 | Learning Rate: 4.9e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   1.05 | Tokens / Sec:   392.1 | Learning Rate: 4.9e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   0.96 | Tokens / Sec:   387.5 | Learning Rate: 4.9e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   0.88 | Tokens / Sec:   384.2 | Learning Rate: 4.9e-04
[mps] Epoch 8 Validation ====
(tensor(1.6422, device='mps:0'), <__main__.TrainState object at 0x1257391c0>)
[mps] Epoch 9 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   0.81 | Tokens / Sec:   604.7 | Learning Rate: 4.9e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   0.77 | Tokens / Sec:   394.7 | Learning Rate: 4.9e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   0.89 | Tokens / Sec:   397.4 | Learning Rate: 4.9e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   0.99 | Tokens / Sec:   388.0 | Learning Rate: 4.9e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   0.76 | Tokens / Sec:   384.3 | Learning Rate: 4.8e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   0.78 | Tokens / Sec:   387.5 | Learning Rate: 4.8e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   0.71 | Tokens / Sec:   384.6 | Learning Rate: 4.8e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   0.69 | Tokens / Sec:   382.5 | Learning Rate: 4.8e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   0.75 | Tokens / Sec:   390.7 | Learning Rate: 4.8e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   0.85 | Tokens / Sec:   386.9 | Learning Rate: 4.8e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   0.71 | Tokens / Sec:   388.7 | Learning Rate: 4.8e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   0.80 | Tokens / Sec:   393.2 | Learning Rate: 4.8e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   0.86 | Tokens / Sec:   378.6 | Learning Rate: 4.8e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   0.72 | Tokens / Sec:   387.0 | Learning Rate: 4.7e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   0.91 | Tokens / Sec:   385.1 | Learning Rate: 4.7e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   0.81 | Tokens / Sec:   393.0 | Learning Rate: 4.7e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   0.65 | Tokens / Sec:   385.7 | Learning Rate: 4.7e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   0.73 | Tokens / Sec:   394.8 | Learning Rate: 4.7e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   0.82 | Tokens / Sec:   394.0 | Learning Rate: 4.7e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   0.74 | Tokens / Sec:   373.1 | Learning Rate: 4.7e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   0.99 | Tokens / Sec:   391.9 | Learning Rate: 4.7e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   0.90 | Tokens / Sec:   384.2 | Learning Rate: 4.7e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   0.95 | Tokens / Sec:   381.7 | Learning Rate: 4.6e-04
[mps] Epoch 9 Validation ====
(tensor(1.6656, device='mps:0'), <__main__.TrainState object at 0x1257391c0>)
```

> - **20 Epochs:** Steady decrease in validation loss with stable learning rate warmup/decay.

```
Train worker process using device: mps
[mps] Epoch 0 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   8.13 | Tokens / Sec:   252.3 | Learning Rate: 5.4e-07
Epoch Step:     41 | Accumulation Step:   5 | Loss:   7.97 | Tokens / Sec:   401.7 | Learning Rate: 1.1e-05
Epoch Step:     81 | Accumulation Step:   9 | Loss:   7.63 | Tokens / Sec:   378.8 | Learning Rate: 2.2e-05
Epoch Step:    121 | Accumulation Step:  13 | Loss:   7.39 | Tokens / Sec:   379.9 | Learning Rate: 3.3e-05
Epoch Step:    161 | Accumulation Step:  17 | Loss:   7.17 | Tokens / Sec:   386.9 | Learning Rate: 4.4e-05
Epoch Step:    201 | Accumulation Step:  21 | Loss:   7.12 | Tokens / Sec:   371.3 | Learning Rate: 5.4e-05
Epoch Step:    241 | Accumulation Step:  25 | Loss:   6.94 | Tokens / Sec:   382.9 | Learning Rate: 6.5e-05
Epoch Step:    281 | Accumulation Step:  29 | Loss:   6.83 | Tokens / Sec:   377.5 | Learning Rate: 7.6e-05
Epoch Step:    321 | Accumulation Step:  33 | Loss:   6.57 | Tokens / Sec:   384.2 | Learning Rate: 8.7e-05
Epoch Step:    361 | Accumulation Step:  37 | Loss:   6.34 | Tokens / Sec:   383.1 | Learning Rate: 9.7e-05
Epoch Step:    401 | Accumulation Step:  41 | Loss:   6.04 | Tokens / Sec:   378.8 | Learning Rate: 1.1e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   5.81 | Tokens / Sec:   380.2 | Learning Rate: 1.2e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   5.54 | Tokens / Sec:   385.3 | Learning Rate: 1.3e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   5.27 | Tokens / Sec:   380.5 | Learning Rate: 1.4e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   5.09 | Tokens / Sec:   381.9 | Learning Rate: 1.5e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   4.81 | Tokens / Sec:   378.5 | Learning Rate: 1.6e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   4.58 | Tokens / Sec:   376.6 | Learning Rate: 1.7e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   4.56 | Tokens / Sec:   378.3 | Learning Rate: 1.8e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   4.36 | Tokens / Sec:   387.9 | Learning Rate: 1.9e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   4.34 | Tokens / Sec:   383.3 | Learning Rate: 2.0e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   4.19 | Tokens / Sec:   384.5 | Learning Rate: 2.2e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   4.28 | Tokens / Sec:   370.8 | Learning Rate: 2.3e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   4.26 | Tokens / Sec:   394.0 | Learning Rate: 2.4e-04
[mps] Epoch 0 Validation ====
(tensor(4.1624, device='mps:0'), <__main__.TrainState object at 0x122ee6120>)
[mps] Epoch 1 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   4.21 | Tokens / Sec:   365.7 | Learning Rate: 2.4e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   4.06 | Tokens / Sec:   376.3 | Learning Rate: 2.6e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   4.09 | Tokens / Sec:   383.6 | Learning Rate: 2.7e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   4.07 | Tokens / Sec:   383.0 | Learning Rate: 2.8e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   4.12 | Tokens / Sec:   377.2 | Learning Rate: 2.9e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   3.85 | Tokens / Sec:   387.6 | Learning Rate: 3.0e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   3.78 | Tokens / Sec:   385.8 | Learning Rate: 3.1e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   3.86 | Tokens / Sec:   374.7 | Learning Rate: 3.2e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   3.74 | Tokens / Sec:   388.0 | Learning Rate: 3.3e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   3.75 | Tokens / Sec:   376.5 | Learning Rate: 3.4e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   3.77 | Tokens / Sec:   389.4 | Learning Rate: 3.5e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   3.59 | Tokens / Sec:   382.3 | Learning Rate: 3.6e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   3.69 | Tokens / Sec:   376.5 | Learning Rate: 3.7e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   3.57 | Tokens / Sec:   379.1 | Learning Rate: 3.8e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   3.31 | Tokens / Sec:   384.0 | Learning Rate: 4.0e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   3.40 | Tokens / Sec:   378.3 | Learning Rate: 4.1e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   3.39 | Tokens / Sec:   376.6 | Learning Rate: 4.2e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   3.25 | Tokens / Sec:   382.1 | Learning Rate: 4.3e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   3.26 | Tokens / Sec:   388.1 | Learning Rate: 4.4e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   3.39 | Tokens / Sec:   384.4 | Learning Rate: 4.5e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   3.53 | Tokens / Sec:   383.7 | Learning Rate: 4.6e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   3.45 | Tokens / Sec:   381.0 | Learning Rate: 4.7e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   3.02 | Tokens / Sec:   395.5 | Learning Rate: 4.8e-04
[mps] Epoch 1 Validation ====
(tensor(3.1572, device='mps:0'), <__main__.TrainState object at 0x122ee6120>)
[mps] Epoch 2 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   3.15 | Tokens / Sec:   422.8 | Learning Rate: 4.9e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   3.22 | Tokens / Sec:   385.1 | Learning Rate: 5.0e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   3.05 | Tokens / Sec:   387.2 | Learning Rate: 5.1e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   3.12 | Tokens / Sec:   385.2 | Learning Rate: 5.2e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   3.10 | Tokens / Sec:   388.0 | Learning Rate: 5.3e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   3.21 | Tokens / Sec:   393.1 | Learning Rate: 5.4e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   3.18 | Tokens / Sec:   385.5 | Learning Rate: 5.5e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   3.11 | Tokens / Sec:   389.7 | Learning Rate: 5.6e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   2.77 | Tokens / Sec:   378.4 | Learning Rate: 5.7e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   3.08 | Tokens / Sec:   389.8 | Learning Rate: 5.9e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   2.67 | Tokens / Sec:   385.5 | Learning Rate: 6.0e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   2.85 | Tokens / Sec:   385.8 | Learning Rate: 6.1e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   2.85 | Tokens / Sec:   370.3 | Learning Rate: 6.2e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   2.60 | Tokens / Sec:   389.7 | Learning Rate: 6.3e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   3.03 | Tokens / Sec:   385.8 | Learning Rate: 6.4e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   2.43 | Tokens / Sec:   385.6 | Learning Rate: 6.5e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   2.49 | Tokens / Sec:   383.9 | Learning Rate: 6.6e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   2.69 | Tokens / Sec:   378.1 | Learning Rate: 6.7e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   2.52 | Tokens / Sec:   383.0 | Learning Rate: 6.8e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   2.34 | Tokens / Sec:   387.1 | Learning Rate: 6.9e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   2.59 | Tokens / Sec:   387.1 | Learning Rate: 7.0e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   2.46 | Tokens / Sec:   386.6 | Learning Rate: 7.1e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   2.80 | Tokens / Sec:   392.5 | Learning Rate: 7.3e-04
[mps] Epoch 2 Validation ====
(tensor(2.4232, device='mps:0'), <__main__.TrainState object at 0x122ee6120>)
[mps] Epoch 3 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   2.33 | Tokens / Sec:   415.2 | Learning Rate: 7.3e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   2.14 | Tokens / Sec:   378.9 | Learning Rate: 7.4e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   2.22 | Tokens / Sec:   381.0 | Learning Rate: 7.5e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   2.11 | Tokens / Sec:   383.1 | Learning Rate: 7.6e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   2.40 | Tokens / Sec:   382.6 | Learning Rate: 7.8e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   2.21 | Tokens / Sec:   379.9 | Learning Rate: 7.9e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   2.21 | Tokens / Sec:   382.6 | Learning Rate: 8.0e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   2.40 | Tokens / Sec:   387.1 | Learning Rate: 8.1e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   1.89 | Tokens / Sec:   380.4 | Learning Rate: 8.0e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   1.98 | Tokens / Sec:   382.9 | Learning Rate: 8.0e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   2.10 | Tokens / Sec:   383.3 | Learning Rate: 7.9e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   2.05 | Tokens / Sec:   384.2 | Learning Rate: 7.9e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   2.08 | Tokens / Sec:   388.1 | Learning Rate: 7.8e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   2.13 | Tokens / Sec:   380.9 | Learning Rate: 7.8e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   1.81 | Tokens / Sec:   383.0 | Learning Rate: 7.7e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   1.80 | Tokens / Sec:   382.3 | Learning Rate: 7.7e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   2.27 | Tokens / Sec:   383.3 | Learning Rate: 7.6e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   1.84 | Tokens / Sec:   384.3 | Learning Rate: 7.6e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   1.98 | Tokens / Sec:   385.6 | Learning Rate: 7.5e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   1.86 | Tokens / Sec:   387.0 | Learning Rate: 7.5e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   2.03 | Tokens / Sec:   386.2 | Learning Rate: 7.4e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   2.18 | Tokens / Sec:   382.8 | Learning Rate: 7.4e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   1.78 | Tokens / Sec:   394.6 | Learning Rate: 7.4e-04
[mps] Epoch 3 Validation ====
(tensor(1.9501, device='mps:0'), <__main__.TrainState object at 0x122ee6120>)
[mps] Epoch 4 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   1.69 | Tokens / Sec:   434.3 | Learning Rate: 7.3e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   1.75 | Tokens / Sec:   387.6 | Learning Rate: 7.3e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   1.51 | Tokens / Sec:   389.7 | Learning Rate: 7.3e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   1.70 | Tokens / Sec:   385.3 | Learning Rate: 7.2e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   1.71 | Tokens / Sec:   392.7 | Learning Rate: 7.2e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   1.60 | Tokens / Sec:   377.4 | Learning Rate: 7.1e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   1.58 | Tokens / Sec:   389.4 | Learning Rate: 7.1e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   1.75 | Tokens / Sec:   370.6 | Learning Rate: 7.1e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   1.51 | Tokens / Sec:   394.3 | Learning Rate: 7.0e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   1.74 | Tokens / Sec:   396.4 | Learning Rate: 7.0e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   1.45 | Tokens / Sec:   383.8 | Learning Rate: 7.0e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   1.74 | Tokens / Sec:   399.3 | Learning Rate: 6.9e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   1.52 | Tokens / Sec:   388.9 | Learning Rate: 6.9e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   1.58 | Tokens / Sec:   393.8 | Learning Rate: 6.9e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   1.43 | Tokens / Sec:   382.3 | Learning Rate: 6.8e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   1.55 | Tokens / Sec:   384.3 | Learning Rate: 6.8e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   1.84 | Tokens / Sec:   381.5 | Learning Rate: 6.8e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   1.60 | Tokens / Sec:   386.0 | Learning Rate: 6.7e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   1.65 | Tokens / Sec:   382.7 | Learning Rate: 6.7e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   1.59 | Tokens / Sec:   384.7 | Learning Rate: 6.7e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   1.59 | Tokens / Sec:   384.8 | Learning Rate: 6.6e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   1.85 | Tokens / Sec:   396.5 | Learning Rate: 6.6e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   1.48 | Tokens / Sec:   401.4 | Learning Rate: 6.6e-04
[mps] Epoch 4 Validation ====
(tensor(1.7691, device='mps:0'), <__main__.TrainState object at 0x122ee6120>)
[mps] Epoch 5 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   1.43 | Tokens / Sec:   430.7 | Learning Rate: 6.6e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   1.23 | Tokens / Sec:   374.1 | Learning Rate: 6.5e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   1.16 | Tokens / Sec:   378.7 | Learning Rate: 6.5e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   1.44 | Tokens / Sec:   383.2 | Learning Rate: 6.5e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   1.02 | Tokens / Sec:   385.2 | Learning Rate: 6.4e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   1.33 | Tokens / Sec:   378.6 | Learning Rate: 6.4e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   1.38 | Tokens / Sec:   376.0 | Learning Rate: 6.4e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   1.20 | Tokens / Sec:   376.0 | Learning Rate: 6.4e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   1.62 | Tokens / Sec:   382.8 | Learning Rate: 6.3e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   1.37 | Tokens / Sec:   381.0 | Learning Rate: 6.3e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   1.45 | Tokens / Sec:   379.2 | Learning Rate: 6.3e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   1.34 | Tokens / Sec:   380.5 | Learning Rate: 6.3e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   1.34 | Tokens / Sec:   382.6 | Learning Rate: 6.2e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   1.09 | Tokens / Sec:   375.2 | Learning Rate: 6.2e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   1.43 | Tokens / Sec:   374.5 | Learning Rate: 6.2e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   1.31 | Tokens / Sec:   376.5 | Learning Rate: 6.2e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   1.12 | Tokens / Sec:   381.9 | Learning Rate: 6.1e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   1.21 | Tokens / Sec:   379.0 | Learning Rate: 6.1e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   1.40 | Tokens / Sec:   383.8 | Learning Rate: 6.1e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   1.35 | Tokens / Sec:   388.6 | Learning Rate: 6.1e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   1.40 | Tokens / Sec:   375.5 | Learning Rate: 6.0e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   1.48 | Tokens / Sec:   381.2 | Learning Rate: 6.0e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   1.44 | Tokens / Sec:   415.4 | Learning Rate: 6.0e-04
[mps] Epoch 5 Validation ====
(tensor(1.6947, device='mps:0'), <__main__.TrainState object at 0x122ee6120>)
[mps] Epoch 6 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   1.21 | Tokens / Sec:   413.2 | Learning Rate: 6.0e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   1.00 | Tokens / Sec:   387.9 | Learning Rate: 6.0e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   1.17 | Tokens / Sec:   380.6 | Learning Rate: 5.9e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   1.23 | Tokens / Sec:   380.0 | Learning Rate: 5.9e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   1.25 | Tokens / Sec:   389.3 | Learning Rate: 5.9e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   0.98 | Tokens / Sec:   386.9 | Learning Rate: 5.9e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   0.95 | Tokens / Sec:   394.2 | Learning Rate: 5.9e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   1.33 | Tokens / Sec:   380.6 | Learning Rate: 5.8e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   1.34 | Tokens / Sec:   396.9 | Learning Rate: 5.8e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   1.16 | Tokens / Sec:   394.3 | Learning Rate: 5.8e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   1.06 | Tokens / Sec:   385.3 | Learning Rate: 5.8e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   0.99 | Tokens / Sec:   394.6 | Learning Rate: 5.8e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   1.10 | Tokens / Sec:   387.6 | Learning Rate: 5.7e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   1.12 | Tokens / Sec:   387.8 | Learning Rate: 5.7e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   1.21 | Tokens / Sec:   387.9 | Learning Rate: 5.7e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   1.26 | Tokens / Sec:   388.5 | Learning Rate: 5.7e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   1.16 | Tokens / Sec:   384.2 | Learning Rate: 5.7e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   1.06 | Tokens / Sec:   389.7 | Learning Rate: 5.6e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   1.10 | Tokens / Sec:   380.0 | Learning Rate: 5.6e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   1.24 | Tokens / Sec:   385.3 | Learning Rate: 5.6e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   0.96 | Tokens / Sec:   374.1 | Learning Rate: 5.6e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   1.11 | Tokens / Sec:   387.6 | Learning Rate: 5.6e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   1.31 | Tokens / Sec:   419.2 | Learning Rate: 5.6e-04
[mps] Epoch 6 Validation ====
(tensor(1.6515, device='mps:0'), <__main__.TrainState object at 0x122ee6120>)
[mps] Epoch 7 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   0.94 | Tokens / Sec:   479.3 | Learning Rate: 5.5e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   1.04 | Tokens / Sec:   385.7 | Learning Rate: 5.5e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   0.92 | Tokens / Sec:   385.9 | Learning Rate: 5.5e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   0.92 | Tokens / Sec:   385.0 | Learning Rate: 5.5e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   0.99 | Tokens / Sec:   371.7 | Learning Rate: 5.5e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   1.27 | Tokens / Sec:   379.9 | Learning Rate: 5.5e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   1.03 | Tokens / Sec:   392.7 | Learning Rate: 5.4e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   0.96 | Tokens / Sec:   390.5 | Learning Rate: 5.4e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   0.99 | Tokens / Sec:   388.2 | Learning Rate: 5.4e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   0.93 | Tokens / Sec:   385.1 | Learning Rate: 5.4e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   1.14 | Tokens / Sec:   385.2 | Learning Rate: 5.4e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   0.91 | Tokens / Sec:   392.9 | Learning Rate: 5.4e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   1.04 | Tokens / Sec:   389.1 | Learning Rate: 5.3e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   1.03 | Tokens / Sec:   386.5 | Learning Rate: 5.3e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   1.11 | Tokens / Sec:   380.3 | Learning Rate: 5.3e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   1.11 | Tokens / Sec:   378.5 | Learning Rate: 5.3e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   0.88 | Tokens / Sec:   387.9 | Learning Rate: 5.3e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   1.14 | Tokens / Sec:   373.5 | Learning Rate: 5.3e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   0.85 | Tokens / Sec:   383.9 | Learning Rate: 5.3e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   0.95 | Tokens / Sec:   384.4 | Learning Rate: 5.2e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   1.01 | Tokens / Sec:   383.2 | Learning Rate: 5.2e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   0.98 | Tokens / Sec:   388.6 | Learning Rate: 5.2e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   0.84 | Tokens / Sec:   416.7 | Learning Rate: 5.2e-04
[mps] Epoch 7 Validation ====
(tensor(1.6581, device='mps:0'), <__main__.TrainState object at 0x122ee6120>)
[mps] Epoch 8 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   0.83 | Tokens / Sec:   446.6 | Learning Rate: 5.2e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   0.82 | Tokens / Sec:   389.3 | Learning Rate: 5.2e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   0.77 | Tokens / Sec:   394.0 | Learning Rate: 5.2e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   0.83 | Tokens / Sec:   386.7 | Learning Rate: 5.1e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   0.82 | Tokens / Sec:   379.0 | Learning Rate: 5.1e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   1.01 | Tokens / Sec:   384.0 | Learning Rate: 5.1e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   0.95 | Tokens / Sec:   386.1 | Learning Rate: 5.1e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   0.80 | Tokens / Sec:   380.7 | Learning Rate: 5.1e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   0.87 | Tokens / Sec:   382.5 | Learning Rate: 5.1e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   1.03 | Tokens / Sec:   383.0 | Learning Rate: 5.1e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   0.95 | Tokens / Sec:   385.7 | Learning Rate: 5.1e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   0.96 | Tokens / Sec:   383.1 | Learning Rate: 5.0e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   0.84 | Tokens / Sec:   392.9 | Learning Rate: 5.0e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   0.88 | Tokens / Sec:   393.5 | Learning Rate: 5.0e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   0.91 | Tokens / Sec:   389.7 | Learning Rate: 5.0e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   0.84 | Tokens / Sec:   380.2 | Learning Rate: 5.0e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   0.93 | Tokens / Sec:   385.2 | Learning Rate: 5.0e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   1.01 | Tokens / Sec:   381.9 | Learning Rate: 5.0e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   0.87 | Tokens / Sec:   378.3 | Learning Rate: 4.9e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   0.89 | Tokens / Sec:   388.0 | Learning Rate: 4.9e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   0.87 | Tokens / Sec:   388.2 | Learning Rate: 4.9e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   0.97 | Tokens / Sec:   399.4 | Learning Rate: 4.9e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   0.83 | Tokens / Sec:   428.3 | Learning Rate: 4.9e-04
[mps] Epoch 8 Validation ====
(tensor(1.6419, device='mps:0'), <__main__.TrainState object at 0x122ee6120>)
[mps] Epoch 9 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   0.76 | Tokens / Sec:   409.6 | Learning Rate: 4.9e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   0.63 | Tokens / Sec:   385.0 | Learning Rate: 4.9e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   0.67 | Tokens / Sec:   375.2 | Learning Rate: 4.9e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   0.89 | Tokens / Sec:   387.0 | Learning Rate: 4.9e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   0.71 | Tokens / Sec:   383.9 | Learning Rate: 4.8e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   1.03 | Tokens / Sec:   394.6 | Learning Rate: 4.8e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   0.84 | Tokens / Sec:   388.3 | Learning Rate: 4.8e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   0.83 | Tokens / Sec:   387.5 | Learning Rate: 4.8e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   0.88 | Tokens / Sec:   386.5 | Learning Rate: 4.8e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   0.81 | Tokens / Sec:   385.6 | Learning Rate: 4.8e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   0.94 | Tokens / Sec:   392.8 | Learning Rate: 4.8e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   0.82 | Tokens / Sec:   379.1 | Learning Rate: 4.8e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   0.76 | Tokens / Sec:   389.6 | Learning Rate: 4.8e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   0.75 | Tokens / Sec:   390.9 | Learning Rate: 4.7e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   0.86 | Tokens / Sec:   388.6 | Learning Rate: 4.7e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   0.86 | Tokens / Sec:   387.9 | Learning Rate: 4.7e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   0.90 | Tokens / Sec:   384.7 | Learning Rate: 4.7e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   0.88 | Tokens / Sec:   395.7 | Learning Rate: 4.7e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   0.84 | Tokens / Sec:   393.9 | Learning Rate: 4.7e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   0.88 | Tokens / Sec:   387.5 | Learning Rate: 4.7e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   0.89 | Tokens / Sec:   388.1 | Learning Rate: 4.7e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   0.81 | Tokens / Sec:   386.4 | Learning Rate: 4.7e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   0.75 | Tokens / Sec:   498.3 | Learning Rate: 4.6e-04
[mps] Epoch 9 Validation ====
(tensor(1.6488, device='mps:0'), <__main__.TrainState object at 0x122ee6120>)
[mps] Epoch 10 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   0.67 | Tokens / Sec:   708.5 | Learning Rate: 4.6e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   0.74 | Tokens / Sec:   593.2 | Learning Rate: 4.6e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   0.71 | Tokens / Sec:   589.6 | Learning Rate: 4.6e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   0.69 | Tokens / Sec:   576.9 | Learning Rate: 4.6e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   0.57 | Tokens / Sec:   580.8 | Learning Rate: 4.6e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   0.66 | Tokens / Sec:   582.7 | Learning Rate: 4.6e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   0.64 | Tokens / Sec:   587.1 | Learning Rate: 4.6e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   0.73 | Tokens / Sec:   578.3 | Learning Rate: 4.6e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   0.66 | Tokens / Sec:   593.6 | Learning Rate: 4.6e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   0.69 | Tokens / Sec:   595.3 | Learning Rate: 4.6e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   0.63 | Tokens / Sec:   596.3 | Learning Rate: 4.5e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   0.69 | Tokens / Sec:   582.2 | Learning Rate: 4.5e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   0.60 | Tokens / Sec:   594.1 | Learning Rate: 4.5e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   0.72 | Tokens / Sec:   580.7 | Learning Rate: 4.5e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   0.74 | Tokens / Sec:   580.0 | Learning Rate: 4.5e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   0.74 | Tokens / Sec:   589.9 | Learning Rate: 4.5e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   0.68 | Tokens / Sec:   584.1 | Learning Rate: 4.5e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   0.69 | Tokens / Sec:   583.1 | Learning Rate: 4.5e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   0.73 | Tokens / Sec:   578.2 | Learning Rate: 4.5e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   0.62 | Tokens / Sec:   585.4 | Learning Rate: 4.5e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   0.64 | Tokens / Sec:   586.4 | Learning Rate: 4.4e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   0.72 | Tokens / Sec:   581.9 | Learning Rate: 4.4e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   0.80 | Tokens / Sec:   602.6 | Learning Rate: 4.4e-04
[mps] Epoch 10 Validation ====
(tensor(1.6687, device='mps:0'), <__main__.TrainState object at 0x122ee6120>)
[mps] Epoch 11 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   0.61 | Tokens / Sec:   695.2 | Learning Rate: 4.4e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   0.67 | Tokens / Sec:   592.3 | Learning Rate: 4.4e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   0.61 | Tokens / Sec:   581.4 | Learning Rate: 4.4e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   0.61 | Tokens / Sec:   600.1 | Learning Rate: 4.4e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   0.59 | Tokens / Sec:   587.9 | Learning Rate: 4.4e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   0.76 | Tokens / Sec:   589.8 | Learning Rate: 4.4e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   0.63 | Tokens / Sec:   585.4 | Learning Rate: 4.4e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   0.62 | Tokens / Sec:   588.4 | Learning Rate: 4.4e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   0.68 | Tokens / Sec:   589.1 | Learning Rate: 4.4e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   0.72 | Tokens / Sec:   589.7 | Learning Rate: 4.3e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   0.54 | Tokens / Sec:   580.6 | Learning Rate: 4.3e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   0.65 | Tokens / Sec:   577.4 | Learning Rate: 4.3e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   0.63 | Tokens / Sec:   582.5 | Learning Rate: 4.3e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   0.69 | Tokens / Sec:   576.0 | Learning Rate: 4.3e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   0.66 | Tokens / Sec:   587.9 | Learning Rate: 4.3e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   0.74 | Tokens / Sec:   579.5 | Learning Rate: 4.3e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   0.63 | Tokens / Sec:   587.4 | Learning Rate: 4.3e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   0.53 | Tokens / Sec:   578.0 | Learning Rate: 4.3e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   0.63 | Tokens / Sec:   581.6 | Learning Rate: 4.3e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   0.59 | Tokens / Sec:   583.5 | Learning Rate: 4.3e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   0.68 | Tokens / Sec:   594.9 | Learning Rate: 4.3e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   0.65 | Tokens / Sec:   583.0 | Learning Rate: 4.2e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   0.70 | Tokens / Sec:   604.2 | Learning Rate: 4.2e-04
[mps] Epoch 11 Validation ====
(tensor(1.6890, device='mps:0'), <__main__.TrainState object at 0x122ee6120>)
[mps] Epoch 12 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   0.63 | Tokens / Sec:   712.9 | Learning Rate: 4.2e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   0.50 | Tokens / Sec:   585.3 | Learning Rate: 4.2e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   0.50 | Tokens / Sec:   592.8 | Learning Rate: 4.2e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   0.51 | Tokens / Sec:   589.4 | Learning Rate: 4.2e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   0.55 | Tokens / Sec:   588.7 | Learning Rate: 4.2e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   0.64 | Tokens / Sec:   581.0 | Learning Rate: 4.2e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   0.60 | Tokens / Sec:   577.4 | Learning Rate: 4.2e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   0.58 | Tokens / Sec:   549.8 | Learning Rate: 4.2e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   0.68 | Tokens / Sec:   571.6 | Learning Rate: 4.2e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   0.52 | Tokens / Sec:   575.4 | Learning Rate: 4.2e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   0.56 | Tokens / Sec:   581.7 | Learning Rate: 4.2e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   0.56 | Tokens / Sec:   577.1 | Learning Rate: 4.2e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   0.54 | Tokens / Sec:   573.1 | Learning Rate: 4.1e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   0.51 | Tokens / Sec:   573.1 | Learning Rate: 4.1e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   0.56 | Tokens / Sec:   578.4 | Learning Rate: 4.1e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   0.63 | Tokens / Sec:   573.0 | Learning Rate: 4.1e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   0.64 | Tokens / Sec:   564.8 | Learning Rate: 4.1e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   0.54 | Tokens / Sec:   566.8 | Learning Rate: 4.1e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   0.59 | Tokens / Sec:   573.6 | Learning Rate: 4.1e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   0.49 | Tokens / Sec:   580.6 | Learning Rate: 4.1e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   0.49 | Tokens / Sec:   581.0 | Learning Rate: 4.1e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   0.64 | Tokens / Sec:   579.0 | Learning Rate: 4.1e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   0.57 | Tokens / Sec:   600.9 | Learning Rate: 4.1e-04
[mps] Epoch 12 Validation ====
(tensor(1.6864, device='mps:0'), <__main__.TrainState object at 0x122ee6120>)
[mps] Epoch 13 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   0.47 | Tokens / Sec:   696.4 | Learning Rate: 4.1e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   0.44 | Tokens / Sec:   585.3 | Learning Rate: 4.1e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   0.47 | Tokens / Sec:   579.3 | Learning Rate: 4.1e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   0.53 | Tokens / Sec:   569.1 | Learning Rate: 4.0e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   0.52 | Tokens / Sec:   576.2 | Learning Rate: 4.0e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   0.51 | Tokens / Sec:   561.3 | Learning Rate: 4.0e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   0.55 | Tokens / Sec:   564.4 | Learning Rate: 4.0e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   0.51 | Tokens / Sec:   564.7 | Learning Rate: 4.0e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   0.51 | Tokens / Sec:   577.1 | Learning Rate: 4.0e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   0.47 | Tokens / Sec:   575.0 | Learning Rate: 4.0e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   0.52 | Tokens / Sec:   565.8 | Learning Rate: 4.0e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   0.49 | Tokens / Sec:   569.5 | Learning Rate: 4.0e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   0.43 | Tokens / Sec:   571.2 | Learning Rate: 4.0e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   0.55 | Tokens / Sec:   577.5 | Learning Rate: 4.0e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   0.50 | Tokens / Sec:   570.9 | Learning Rate: 4.0e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   0.54 | Tokens / Sec:   574.3 | Learning Rate: 4.0e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   0.45 | Tokens / Sec:   570.4 | Learning Rate: 4.0e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   0.53 | Tokens / Sec:   573.9 | Learning Rate: 4.0e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   0.52 | Tokens / Sec:   571.6 | Learning Rate: 4.0e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   0.50 | Tokens / Sec:   571.9 | Learning Rate: 3.9e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   0.44 | Tokens / Sec:   572.0 | Learning Rate: 3.9e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   0.52 | Tokens / Sec:   571.9 | Learning Rate: 3.9e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   0.49 | Tokens / Sec:   579.0 | Learning Rate: 3.9e-04
[mps] Epoch 13 Validation ====
(tensor(1.7117, device='mps:0'), <__main__.TrainState object at 0x122ee6120>)
[mps] Epoch 14 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   0.46 | Tokens / Sec:   691.3 | Learning Rate: 3.9e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   0.45 | Tokens / Sec:   577.8 | Learning Rate: 3.9e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   0.43 | Tokens / Sec:   564.6 | Learning Rate: 3.9e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   0.42 | Tokens / Sec:   575.5 | Learning Rate: 3.9e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   0.51 | Tokens / Sec:   571.3 | Learning Rate: 3.9e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   0.48 | Tokens / Sec:   576.4 | Learning Rate: 3.9e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   0.43 | Tokens / Sec:   561.3 | Learning Rate: 3.9e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   0.47 | Tokens / Sec:   570.1 | Learning Rate: 3.9e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   0.38 | Tokens / Sec:   576.8 | Learning Rate: 3.9e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   0.53 | Tokens / Sec:   567.4 | Learning Rate: 3.9e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   0.43 | Tokens / Sec:   573.5 | Learning Rate: 3.9e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   0.46 | Tokens / Sec:   575.0 | Learning Rate: 3.9e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   0.40 | Tokens / Sec:   559.8 | Learning Rate: 3.8e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   0.46 | Tokens / Sec:   579.4 | Learning Rate: 3.8e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   0.48 | Tokens / Sec:   576.8 | Learning Rate: 3.8e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   0.44 | Tokens / Sec:   573.4 | Learning Rate: 3.8e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   0.58 | Tokens / Sec:   563.5 | Learning Rate: 3.8e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   0.45 | Tokens / Sec:   569.3 | Learning Rate: 3.8e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   0.45 | Tokens / Sec:   562.1 | Learning Rate: 3.8e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   0.46 | Tokens / Sec:   564.3 | Learning Rate: 3.8e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   0.45 | Tokens / Sec:   568.6 | Learning Rate: 3.8e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   0.49 | Tokens / Sec:   567.5 | Learning Rate: 3.8e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   0.51 | Tokens / Sec:   585.7 | Learning Rate: 3.8e-04
[mps] Epoch 14 Validation ====
(tensor(1.7565, device='mps:0'), <__main__.TrainState object at 0x122ee6120>)
[mps] Epoch 15 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   0.45 | Tokens / Sec:   728.8 | Learning Rate: 3.8e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   0.42 | Tokens / Sec:   572.2 | Learning Rate: 3.8e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   0.34 | Tokens / Sec:   568.9 | Learning Rate: 3.8e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   0.38 | Tokens / Sec:   565.7 | Learning Rate: 3.8e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   0.40 | Tokens / Sec:   570.2 | Learning Rate: 3.8e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   0.51 | Tokens / Sec:   567.7 | Learning Rate: 3.8e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   0.41 | Tokens / Sec:   565.0 | Learning Rate: 3.8e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   0.42 | Tokens / Sec:   585.7 | Learning Rate: 3.8e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   0.46 | Tokens / Sec:   566.6 | Learning Rate: 3.7e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   0.43 | Tokens / Sec:   567.5 | Learning Rate: 3.7e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   0.34 | Tokens / Sec:   573.3 | Learning Rate: 3.7e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   0.44 | Tokens / Sec:   570.9 | Learning Rate: 3.7e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   0.42 | Tokens / Sec:   574.2 | Learning Rate: 3.7e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   0.40 | Tokens / Sec:   577.0 | Learning Rate: 3.7e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   0.43 | Tokens / Sec:   570.3 | Learning Rate: 3.7e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   0.50 | Tokens / Sec:   577.0 | Learning Rate: 3.7e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   0.44 | Tokens / Sec:   579.5 | Learning Rate: 3.7e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   0.37 | Tokens / Sec:   576.3 | Learning Rate: 3.7e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   0.39 | Tokens / Sec:   578.9 | Learning Rate: 3.7e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   0.45 | Tokens / Sec:   567.1 | Learning Rate: 3.7e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   0.39 | Tokens / Sec:   562.4 | Learning Rate: 3.7e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   0.43 | Tokens / Sec:   574.8 | Learning Rate: 3.7e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   0.47 | Tokens / Sec:   590.3 | Learning Rate: 3.7e-04
[mps] Epoch 15 Validation ====
(tensor(1.7519, device='mps:0'), <__main__.TrainState object at 0x122ee6120>)
[mps] Epoch 16 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   0.39 | Tokens / Sec:   681.6 | Learning Rate: 3.7e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   0.37 | Tokens / Sec:   574.6 | Learning Rate: 3.7e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   0.38 | Tokens / Sec:   574.7 | Learning Rate: 3.7e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   0.34 | Tokens / Sec:   562.4 | Learning Rate: 3.7e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   0.36 | Tokens / Sec:   577.2 | Learning Rate: 3.6e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   0.31 | Tokens / Sec:   563.4 | Learning Rate: 3.6e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   0.41 | Tokens / Sec:   576.6 | Learning Rate: 3.6e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   0.35 | Tokens / Sec:   575.8 | Learning Rate: 3.6e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   0.38 | Tokens / Sec:   580.3 | Learning Rate: 3.6e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   0.40 | Tokens / Sec:   570.4 | Learning Rate: 3.6e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   0.38 | Tokens / Sec:   583.3 | Learning Rate: 3.6e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   0.40 | Tokens / Sec:   566.8 | Learning Rate: 3.6e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   0.40 | Tokens / Sec:   575.3 | Learning Rate: 3.6e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   0.35 | Tokens / Sec:   559.3 | Learning Rate: 3.6e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   0.42 | Tokens / Sec:   574.2 | Learning Rate: 3.6e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   0.45 | Tokens / Sec:   572.0 | Learning Rate: 3.6e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   0.44 | Tokens / Sec:   579.7 | Learning Rate: 3.6e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   0.42 | Tokens / Sec:   568.2 | Learning Rate: 3.6e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   0.32 | Tokens / Sec:   567.9 | Learning Rate: 3.6e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   0.38 | Tokens / Sec:   567.2 | Learning Rate: 3.6e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   0.39 | Tokens / Sec:   561.0 | Learning Rate: 3.6e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   0.40 | Tokens / Sec:   575.0 | Learning Rate: 3.6e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   0.39 | Tokens / Sec:   583.4 | Learning Rate: 3.6e-04
[mps] Epoch 16 Validation ====
(tensor(1.7793, device='mps:0'), <__main__.TrainState object at 0x122ee6120>)
[mps] Epoch 17 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   0.40 | Tokens / Sec:   668.9 | Learning Rate: 3.6e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   0.32 | Tokens / Sec:   562.4 | Learning Rate: 3.6e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   0.33 | Tokens / Sec:   564.3 | Learning Rate: 3.5e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   0.34 | Tokens / Sec:   561.6 | Learning Rate: 3.5e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   0.36 | Tokens / Sec:   571.3 | Learning Rate: 3.5e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   0.36 | Tokens / Sec:   567.6 | Learning Rate: 3.5e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   0.43 | Tokens / Sec:   565.0 | Learning Rate: 3.5e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   0.36 | Tokens / Sec:   581.1 | Learning Rate: 3.5e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   0.33 | Tokens / Sec:   576.0 | Learning Rate: 3.5e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   0.38 | Tokens / Sec:   562.8 | Learning Rate: 3.5e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   0.33 | Tokens / Sec:   563.3 | Learning Rate: 3.5e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   0.37 | Tokens / Sec:   566.5 | Learning Rate: 3.5e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   0.34 | Tokens / Sec:   574.6 | Learning Rate: 3.5e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   0.30 | Tokens / Sec:   563.8 | Learning Rate: 3.5e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   0.37 | Tokens / Sec:   556.5 | Learning Rate: 3.5e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   0.39 | Tokens / Sec:   578.9 | Learning Rate: 3.5e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   0.41 | Tokens / Sec:   570.3 | Learning Rate: 3.5e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   0.35 | Tokens / Sec:   572.7 | Learning Rate: 3.5e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   0.40 | Tokens / Sec:   571.4 | Learning Rate: 3.5e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   0.32 | Tokens / Sec:   566.3 | Learning Rate: 3.5e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   0.34 | Tokens / Sec:   571.7 | Learning Rate: 3.5e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   0.38 | Tokens / Sec:   561.3 | Learning Rate: 3.5e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   0.39 | Tokens / Sec:   584.6 | Learning Rate: 3.5e-04
[mps] Epoch 17 Validation ====
(tensor(1.7840, device='mps:0'), <__main__.TrainState object at 0x122ee6120>)
[mps] Epoch 18 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   0.36 | Tokens / Sec:   551.5 | Learning Rate: 3.5e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   0.31 | Tokens / Sec:   567.4 | Learning Rate: 3.5e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   0.30 | Tokens / Sec:   560.2 | Learning Rate: 3.5e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   0.32 | Tokens / Sec:   577.9 | Learning Rate: 3.4e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   0.33 | Tokens / Sec:   574.9 | Learning Rate: 3.4e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   0.36 | Tokens / Sec:   573.8 | Learning Rate: 3.4e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   0.30 | Tokens / Sec:   566.3 | Learning Rate: 3.4e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   0.37 | Tokens / Sec:   563.2 | Learning Rate: 3.4e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   0.31 | Tokens / Sec:   559.4 | Learning Rate: 3.4e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   0.34 | Tokens / Sec:   578.0 | Learning Rate: 3.4e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   0.34 | Tokens / Sec:   561.8 | Learning Rate: 3.4e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   0.30 | Tokens / Sec:   570.3 | Learning Rate: 3.4e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   0.31 | Tokens / Sec:   578.2 | Learning Rate: 3.4e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   0.31 | Tokens / Sec:   560.4 | Learning Rate: 3.4e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   0.31 | Tokens / Sec:   573.3 | Learning Rate: 3.4e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   0.36 | Tokens / Sec:   568.1 | Learning Rate: 3.4e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   0.33 | Tokens / Sec:   571.8 | Learning Rate: 3.4e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   0.34 | Tokens / Sec:   574.0 | Learning Rate: 3.4e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   0.39 | Tokens / Sec:   573.3 | Learning Rate: 3.4e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   0.37 | Tokens / Sec:   573.1 | Learning Rate: 3.4e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   0.29 | Tokens / Sec:   576.7 | Learning Rate: 3.4e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   0.31 | Tokens / Sec:   562.7 | Learning Rate: 3.4e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   0.29 | Tokens / Sec:   584.5 | Learning Rate: 3.4e-04
[mps] Epoch 18 Validation ====
(tensor(1.8159, device='mps:0'), <__main__.TrainState object at 0x122ee6120>)
[mps] Epoch 19 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   0.25 | Tokens / Sec:   675.1 | Learning Rate: 3.4e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   0.32 | Tokens / Sec:   567.2 | Learning Rate: 3.4e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   0.30 | Tokens / Sec:   570.8 | Learning Rate: 3.4e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   0.29 | Tokens / Sec:   564.0 | Learning Rate: 3.4e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   0.31 | Tokens / Sec:   580.8 | Learning Rate: 3.4e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   0.24 | Tokens / Sec:   576.2 | Learning Rate: 3.3e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   0.29 | Tokens / Sec:   572.8 | Learning Rate: 3.3e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   0.32 | Tokens / Sec:   571.7 | Learning Rate: 3.3e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   0.29 | Tokens / Sec:   567.6 | Learning Rate: 3.3e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   0.29 | Tokens / Sec:   575.8 | Learning Rate: 3.3e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   0.36 | Tokens / Sec:   580.7 | Learning Rate: 3.3e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   0.29 | Tokens / Sec:   564.1 | Learning Rate: 3.3e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   0.29 | Tokens / Sec:   552.2 | Learning Rate: 3.3e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   0.29 | Tokens / Sec:   557.2 | Learning Rate: 3.3e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   0.36 | Tokens / Sec:   564.9 | Learning Rate: 3.3e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   0.30 | Tokens / Sec:   561.4 | Learning Rate: 3.3e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   0.34 | Tokens / Sec:   577.5 | Learning Rate: 3.3e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   0.29 | Tokens / Sec:   586.3 | Learning Rate: 3.3e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   0.32 | Tokens / Sec:   575.3 | Learning Rate: 3.3e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   0.28 | Tokens / Sec:   568.4 | Learning Rate: 3.3e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   0.34 | Tokens / Sec:   562.4 | Learning Rate: 3.3e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   0.30 | Tokens / Sec:   566.2 | Learning Rate: 3.3e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   0.31 | Tokens / Sec:   568.0 | Learning Rate: 3.3e-04
[mps] Epoch 19 Validation ====
(tensor(1.8305, device='mps:0'), <__main__.TrainState object at 0x122ee6120>)
```

> - **30 Epochs:** Optimal convergence for the synthetic task with near-zero loss and high sequence generation accuracy.

```
Train worker process using device: mps
[mps] Epoch 0 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   8.11 | Tokens / Sec:   227.7 | Learning Rate: 5.4e-07
Epoch Step:     41 | Accumulation Step:   5 | Loss:   7.93 | Tokens / Sec:   386.5 | Learning Rate: 1.1e-05
Epoch Step:     81 | Accumulation Step:   9 | Loss:   7.60 | Tokens / Sec:   384.5 | Learning Rate: 2.2e-05
Epoch Step:    121 | Accumulation Step:  13 | Loss:   7.40 | Tokens / Sec:   384.4 | Learning Rate: 3.3e-05
Epoch Step:    161 | Accumulation Step:  17 | Loss:   7.18 | Tokens / Sec:   383.9 | Learning Rate: 4.4e-05
Epoch Step:    201 | Accumulation Step:  21 | Loss:   7.04 | Tokens / Sec:   388.1 | Learning Rate: 5.4e-05
Epoch Step:    241 | Accumulation Step:  25 | Loss:   6.96 | Tokens / Sec:   391.5 | Learning Rate: 6.5e-05
Epoch Step:    281 | Accumulation Step:  29 | Loss:   6.74 | Tokens / Sec:   391.1 | Learning Rate: 7.6e-05
Epoch Step:    321 | Accumulation Step:  33 | Loss:   6.54 | Tokens / Sec:   385.9 | Learning Rate: 8.7e-05
Epoch Step:    361 | Accumulation Step:  37 | Loss:   6.24 | Tokens / Sec:   392.2 | Learning Rate: 9.7e-05
Epoch Step:    401 | Accumulation Step:  41 | Loss:   6.07 | Tokens / Sec:   377.5 | Learning Rate: 1.1e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   5.80 | Tokens / Sec:   385.7 | Learning Rate: 1.2e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   5.61 | Tokens / Sec:   385.5 | Learning Rate: 1.3e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   5.37 | Tokens / Sec:   388.2 | Learning Rate: 1.4e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   4.98 | Tokens / Sec:   380.4 | Learning Rate: 1.5e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   4.94 | Tokens / Sec:   389.2 | Learning Rate: 1.6e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   4.79 | Tokens / Sec:   386.0 | Learning Rate: 1.7e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   4.44 | Tokens / Sec:   386.5 | Learning Rate: 1.8e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   4.46 | Tokens / Sec:   380.6 | Learning Rate: 1.9e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   4.35 | Tokens / Sec:   381.3 | Learning Rate: 2.0e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   4.35 | Tokens / Sec:   383.6 | Learning Rate: 2.2e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   4.30 | Tokens / Sec:   380.4 | Learning Rate: 2.3e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   4.19 | Tokens / Sec:   397.2 | Learning Rate: 2.4e-04
[mps] Epoch 0 Validation ====
(tensor(4.1653, device='mps:0'), <__main__.TrainState object at 0x123462840>)
[mps] Epoch 1 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   3.97 | Tokens / Sec:   458.5 | Learning Rate: 2.4e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   3.98 | Tokens / Sec:   385.7 | Learning Rate: 2.6e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   3.96 | Tokens / Sec:   376.6 | Learning Rate: 2.7e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   4.27 | Tokens / Sec:   395.9 | Learning Rate: 2.8e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   4.06 | Tokens / Sec:   380.2 | Learning Rate: 2.9e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   3.83 | Tokens / Sec:   378.5 | Learning Rate: 3.0e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   3.97 | Tokens / Sec:   381.5 | Learning Rate: 3.1e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   3.58 | Tokens / Sec:   377.0 | Learning Rate: 3.2e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   3.76 | Tokens / Sec:   388.4 | Learning Rate: 3.3e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   3.69 | Tokens / Sec:   377.4 | Learning Rate: 3.4e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   3.41 | Tokens / Sec:   380.0 | Learning Rate: 3.5e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   3.48 | Tokens / Sec:   382.4 | Learning Rate: 3.6e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   3.53 | Tokens / Sec:   385.7 | Learning Rate: 3.7e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   3.38 | Tokens / Sec:   375.3 | Learning Rate: 3.8e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   3.50 | Tokens / Sec:   377.4 | Learning Rate: 4.0e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   3.34 | Tokens / Sec:   382.0 | Learning Rate: 4.1e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   3.37 | Tokens / Sec:   390.8 | Learning Rate: 4.2e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   3.30 | Tokens / Sec:   383.4 | Learning Rate: 4.3e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   3.59 | Tokens / Sec:   384.9 | Learning Rate: 4.4e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   3.34 | Tokens / Sec:   390.0 | Learning Rate: 4.5e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   3.21 | Tokens / Sec:   381.7 | Learning Rate: 4.6e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   3.08 | Tokens / Sec:   396.4 | Learning Rate: 4.7e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   3.47 | Tokens / Sec:   400.2 | Learning Rate: 4.8e-04
[mps] Epoch 1 Validation ====
(tensor(3.1327, device='mps:0'), <__main__.TrainState object at 0x123462840>)
[mps] Epoch 2 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   2.92 | Tokens / Sec:   446.4 | Learning Rate: 4.9e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   3.17 | Tokens / Sec:   385.3 | Learning Rate: 5.0e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   3.10 | Tokens / Sec:   385.6 | Learning Rate: 5.1e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   3.07 | Tokens / Sec:   399.2 | Learning Rate: 5.2e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   2.95 | Tokens / Sec:   384.8 | Learning Rate: 5.3e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   2.92 | Tokens / Sec:   382.6 | Learning Rate: 5.4e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   2.76 | Tokens / Sec:   387.0 | Learning Rate: 5.5e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   3.05 | Tokens / Sec:   375.7 | Learning Rate: 5.6e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   3.06 | Tokens / Sec:   385.0 | Learning Rate: 5.7e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   2.88 | Tokens / Sec:   389.8 | Learning Rate: 5.9e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   2.75 | Tokens / Sec:   382.3 | Learning Rate: 6.0e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   2.64 | Tokens / Sec:   385.4 | Learning Rate: 6.1e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   2.67 | Tokens / Sec:   392.7 | Learning Rate: 6.2e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   2.44 | Tokens / Sec:   381.2 | Learning Rate: 6.3e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   2.65 | Tokens / Sec:   380.6 | Learning Rate: 6.4e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   2.51 | Tokens / Sec:   388.0 | Learning Rate: 6.5e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   2.53 | Tokens / Sec:   386.7 | Learning Rate: 6.6e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   2.48 | Tokens / Sec:   391.6 | Learning Rate: 6.7e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   2.65 | Tokens / Sec:   385.6 | Learning Rate: 6.8e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   2.49 | Tokens / Sec:   383.8 | Learning Rate: 6.9e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   2.32 | Tokens / Sec:   382.1 | Learning Rate: 7.0e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   2.46 | Tokens / Sec:   391.3 | Learning Rate: 7.1e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   2.25 | Tokens / Sec:   390.7 | Learning Rate: 7.3e-04
[mps] Epoch 2 Validation ====
(tensor(2.3427, device='mps:0'), <__main__.TrainState object at 0x123462840>)
[mps] Epoch 3 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   2.36 | Tokens / Sec:   414.2 | Learning Rate: 7.3e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   2.02 | Tokens / Sec:   395.0 | Learning Rate: 7.4e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   2.01 | Tokens / Sec:   377.9 | Learning Rate: 7.5e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   2.21 | Tokens / Sec:   378.5 | Learning Rate: 7.6e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   2.16 | Tokens / Sec:   393.0 | Learning Rate: 7.8e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   2.03 | Tokens / Sec:   383.7 | Learning Rate: 7.9e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   2.23 | Tokens / Sec:   395.8 | Learning Rate: 8.0e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   2.41 | Tokens / Sec:   390.4 | Learning Rate: 8.1e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   2.24 | Tokens / Sec:   397.5 | Learning Rate: 8.0e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   2.14 | Tokens / Sec:   391.8 | Learning Rate: 8.0e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   2.23 | Tokens / Sec:   383.0 | Learning Rate: 7.9e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   2.09 | Tokens / Sec:   388.2 | Learning Rate: 7.9e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   1.74 | Tokens / Sec:   390.4 | Learning Rate: 7.8e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   1.90 | Tokens / Sec:   389.1 | Learning Rate: 7.8e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   2.29 | Tokens / Sec:   395.7 | Learning Rate: 7.7e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   1.78 | Tokens / Sec:   383.5 | Learning Rate: 7.7e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   2.01 | Tokens / Sec:   379.3 | Learning Rate: 7.6e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   2.12 | Tokens / Sec:   395.1 | Learning Rate: 7.6e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   1.73 | Tokens / Sec:   383.5 | Learning Rate: 7.5e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   2.12 | Tokens / Sec:   386.7 | Learning Rate: 7.5e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   1.65 | Tokens / Sec:   394.1 | Learning Rate: 7.4e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   1.94 | Tokens / Sec:   393.5 | Learning Rate: 7.4e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   2.08 | Tokens / Sec:   392.2 | Learning Rate: 7.4e-04
[mps] Epoch 3 Validation ====
(tensor(1.9300, device='mps:0'), <__main__.TrainState object at 0x123462840>)
[mps] Epoch 4 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   1.67 | Tokens / Sec:   455.2 | Learning Rate: 7.3e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   1.47 | Tokens / Sec:   386.6 | Learning Rate: 7.3e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   1.63 | Tokens / Sec:   383.8 | Learning Rate: 7.3e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   1.76 | Tokens / Sec:   383.6 | Learning Rate: 7.2e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   1.43 | Tokens / Sec:   392.5 | Learning Rate: 7.2e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   1.75 | Tokens / Sec:   392.8 | Learning Rate: 7.1e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   1.73 | Tokens / Sec:   383.3 | Learning Rate: 7.1e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   1.41 | Tokens / Sec:   381.9 | Learning Rate: 7.1e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   1.62 | Tokens / Sec:   383.6 | Learning Rate: 7.0e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   1.78 | Tokens / Sec:   380.0 | Learning Rate: 7.0e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   1.70 | Tokens / Sec:   384.3 | Learning Rate: 7.0e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   1.54 | Tokens / Sec:   387.1 | Learning Rate: 6.9e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   1.54 | Tokens / Sec:   386.7 | Learning Rate: 6.9e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   1.38 | Tokens / Sec:   381.2 | Learning Rate: 6.9e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   1.45 | Tokens / Sec:   377.5 | Learning Rate: 6.8e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   1.53 | Tokens / Sec:   391.8 | Learning Rate: 6.8e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   1.43 | Tokens / Sec:   374.1 | Learning Rate: 6.8e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   1.51 | Tokens / Sec:   379.1 | Learning Rate: 6.7e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   1.58 | Tokens / Sec:   381.3 | Learning Rate: 6.7e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   1.34 | Tokens / Sec:   378.8 | Learning Rate: 6.7e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   1.76 | Tokens / Sec:   389.6 | Learning Rate: 6.6e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   1.56 | Tokens / Sec:   374.2 | Learning Rate: 6.6e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   1.49 | Tokens / Sec:   392.8 | Learning Rate: 6.6e-04
[mps] Epoch 4 Validation ====
(tensor(1.7601, device='mps:0'), <__main__.TrainState object at 0x123462840>)
[mps] Epoch 5 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   1.35 | Tokens / Sec:   531.5 | Learning Rate: 6.6e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   1.52 | Tokens / Sec:   400.3 | Learning Rate: 6.5e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   1.42 | Tokens / Sec:   389.7 | Learning Rate: 6.5e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   1.32 | Tokens / Sec:   396.9 | Learning Rate: 6.5e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   1.42 | Tokens / Sec:   387.5 | Learning Rate: 6.4e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   1.25 | Tokens / Sec:   396.2 | Learning Rate: 6.4e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   1.39 | Tokens / Sec:   382.4 | Learning Rate: 6.4e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   1.45 | Tokens / Sec:   388.9 | Learning Rate: 6.4e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   1.63 | Tokens / Sec:   392.5 | Learning Rate: 6.3e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   1.47 | Tokens / Sec:   391.1 | Learning Rate: 6.3e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   1.17 | Tokens / Sec:   386.0 | Learning Rate: 6.3e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   1.32 | Tokens / Sec:   392.5 | Learning Rate: 6.3e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   1.44 | Tokens / Sec:   389.9 | Learning Rate: 6.2e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   1.27 | Tokens / Sec:   389.7 | Learning Rate: 6.2e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   1.41 | Tokens / Sec:   392.6 | Learning Rate: 6.2e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   1.51 | Tokens / Sec:   380.8 | Learning Rate: 6.2e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   1.39 | Tokens / Sec:   385.9 | Learning Rate: 6.1e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   1.60 | Tokens / Sec:   395.9 | Learning Rate: 6.1e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   1.46 | Tokens / Sec:   386.6 | Learning Rate: 6.1e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   1.37 | Tokens / Sec:   391.2 | Learning Rate: 6.1e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   1.45 | Tokens / Sec:   392.0 | Learning Rate: 6.0e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   1.42 | Tokens / Sec:   392.3 | Learning Rate: 6.0e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   1.15 | Tokens / Sec:   399.6 | Learning Rate: 6.0e-04
[mps] Epoch 5 Validation ====
(tensor(1.6826, device='mps:0'), <__main__.TrainState object at 0x123462840>)
[mps] Epoch 6 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   1.19 | Tokens / Sec:   478.5 | Learning Rate: 6.0e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   1.22 | Tokens / Sec:   403.9 | Learning Rate: 6.0e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   1.20 | Tokens / Sec:   386.1 | Learning Rate: 5.9e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   1.14 | Tokens / Sec:   380.1 | Learning Rate: 5.9e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   0.90 | Tokens / Sec:   379.9 | Learning Rate: 5.9e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   1.00 | Tokens / Sec:   388.0 | Learning Rate: 5.9e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   1.28 | Tokens / Sec:   390.0 | Learning Rate: 5.9e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   1.13 | Tokens / Sec:   385.0 | Learning Rate: 5.8e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   1.18 | Tokens / Sec:   390.2 | Learning Rate: 5.8e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   1.22 | Tokens / Sec:   390.1 | Learning Rate: 5.8e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   1.18 | Tokens / Sec:   385.7 | Learning Rate: 5.8e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   1.28 | Tokens / Sec:   390.3 | Learning Rate: 5.8e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   1.21 | Tokens / Sec:   386.3 | Learning Rate: 5.7e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   1.03 | Tokens / Sec:   381.3 | Learning Rate: 5.7e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   1.21 | Tokens / Sec:   386.3 | Learning Rate: 5.7e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   1.18 | Tokens / Sec:   382.0 | Learning Rate: 5.7e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   1.20 | Tokens / Sec:   397.5 | Learning Rate: 5.7e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   1.32 | Tokens / Sec:   395.4 | Learning Rate: 5.6e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   1.24 | Tokens / Sec:   392.3 | Learning Rate: 5.6e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   1.15 | Tokens / Sec:   383.2 | Learning Rate: 5.6e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   1.09 | Tokens / Sec:   385.6 | Learning Rate: 5.6e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   1.15 | Tokens / Sec:   375.1 | Learning Rate: 5.6e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   1.24 | Tokens / Sec:   385.9 | Learning Rate: 5.6e-04
[mps] Epoch 6 Validation ====
(tensor(1.6498, device='mps:0'), <__main__.TrainState object at 0x123462840>)
[mps] Epoch 7 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   0.98 | Tokens / Sec:   382.6 | Learning Rate: 5.5e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   0.88 | Tokens / Sec:   402.6 | Learning Rate: 5.5e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   1.00 | Tokens / Sec:   387.8 | Learning Rate: 5.5e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   0.96 | Tokens / Sec:   383.5 | Learning Rate: 5.5e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   1.10 | Tokens / Sec:   386.0 | Learning Rate: 5.5e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   0.97 | Tokens / Sec:   388.1 | Learning Rate: 5.5e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   0.91 | Tokens / Sec:   376.3 | Learning Rate: 5.4e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   1.01 | Tokens / Sec:   386.1 | Learning Rate: 5.4e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   0.99 | Tokens / Sec:   384.1 | Learning Rate: 5.4e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   0.95 | Tokens / Sec:   386.5 | Learning Rate: 5.4e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   1.01 | Tokens / Sec:   384.6 | Learning Rate: 5.4e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   1.04 | Tokens / Sec:   390.8 | Learning Rate: 5.4e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   1.01 | Tokens / Sec:   382.4 | Learning Rate: 5.3e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   1.12 | Tokens / Sec:   388.3 | Learning Rate: 5.3e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   1.11 | Tokens / Sec:   390.0 | Learning Rate: 5.3e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   0.98 | Tokens / Sec:   384.8 | Learning Rate: 5.3e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   1.10 | Tokens / Sec:   397.8 | Learning Rate: 5.3e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   1.02 | Tokens / Sec:   384.2 | Learning Rate: 5.3e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   1.02 | Tokens / Sec:   380.9 | Learning Rate: 5.3e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   1.07 | Tokens / Sec:   389.1 | Learning Rate: 5.2e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   1.05 | Tokens / Sec:   383.3 | Learning Rate: 5.2e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   0.93 | Tokens / Sec:   392.0 | Learning Rate: 5.2e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   1.01 | Tokens / Sec:   385.2 | Learning Rate: 5.2e-04
[mps] Epoch 7 Validation ====
(tensor(1.6454, device='mps:0'), <__main__.TrainState object at 0x123462840>)
[mps] Epoch 8 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   0.86 | Tokens / Sec:   486.4 | Learning Rate: 5.2e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   0.87 | Tokens / Sec:   402.7 | Learning Rate: 5.2e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   0.79 | Tokens / Sec:   381.2 | Learning Rate: 5.2e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   0.87 | Tokens / Sec:   384.7 | Learning Rate: 5.1e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   0.93 | Tokens / Sec:   386.2 | Learning Rate: 5.1e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   0.86 | Tokens / Sec:   393.0 | Learning Rate: 5.1e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   0.98 | Tokens / Sec:   390.8 | Learning Rate: 5.1e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   0.96 | Tokens / Sec:   391.9 | Learning Rate: 5.1e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   0.79 | Tokens / Sec:   398.3 | Learning Rate: 5.1e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   0.93 | Tokens / Sec:   393.4 | Learning Rate: 5.1e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   0.90 | Tokens / Sec:   389.9 | Learning Rate: 5.1e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   0.97 | Tokens / Sec:   389.2 | Learning Rate: 5.0e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   0.81 | Tokens / Sec:   392.7 | Learning Rate: 5.0e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   1.10 | Tokens / Sec:   385.9 | Learning Rate: 5.0e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   0.80 | Tokens / Sec:   388.3 | Learning Rate: 5.0e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   0.98 | Tokens / Sec:   388.4 | Learning Rate: 5.0e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   0.94 | Tokens / Sec:   398.4 | Learning Rate: 5.0e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   0.98 | Tokens / Sec:   384.8 | Learning Rate: 5.0e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   0.87 | Tokens / Sec:   385.0 | Learning Rate: 4.9e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   1.04 | Tokens / Sec:   401.1 | Learning Rate: 4.9e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   0.86 | Tokens / Sec:   384.8 | Learning Rate: 4.9e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   0.80 | Tokens / Sec:   388.4 | Learning Rate: 4.9e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   0.98 | Tokens / Sec:   387.3 | Learning Rate: 4.9e-04
[mps] Epoch 8 Validation ====
(tensor(1.6357, device='mps:0'), <__main__.TrainState object at 0x123462840>)
[mps] Epoch 9 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   0.76 | Tokens / Sec:   391.4 | Learning Rate: 4.9e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   0.80 | Tokens / Sec:   394.2 | Learning Rate: 4.9e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   0.73 | Tokens / Sec:   390.2 | Learning Rate: 4.9e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   0.67 | Tokens / Sec:   398.3 | Learning Rate: 4.9e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   0.88 | Tokens / Sec:   386.5 | Learning Rate: 4.8e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   0.74 | Tokens / Sec:   391.2 | Learning Rate: 4.8e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   0.78 | Tokens / Sec:   380.4 | Learning Rate: 4.8e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   0.76 | Tokens / Sec:   382.5 | Learning Rate: 4.8e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   0.84 | Tokens / Sec:   390.7 | Learning Rate: 4.8e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   0.67 | Tokens / Sec:   381.2 | Learning Rate: 4.8e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   0.86 | Tokens / Sec:   390.6 | Learning Rate: 4.8e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   0.73 | Tokens / Sec:   383.3 | Learning Rate: 4.8e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   0.82 | Tokens / Sec:   384.9 | Learning Rate: 4.8e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   0.73 | Tokens / Sec:   381.2 | Learning Rate: 4.7e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   0.95 | Tokens / Sec:   377.6 | Learning Rate: 4.7e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   0.77 | Tokens / Sec:   377.5 | Learning Rate: 4.7e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   0.82 | Tokens / Sec:   382.6 | Learning Rate: 4.7e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   0.99 | Tokens / Sec:   383.7 | Learning Rate: 4.7e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   0.84 | Tokens / Sec:   379.0 | Learning Rate: 4.7e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   0.78 | Tokens / Sec:   383.7 | Learning Rate: 4.7e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   0.98 | Tokens / Sec:   393.5 | Learning Rate: 4.7e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   0.87 | Tokens / Sec:   391.7 | Learning Rate: 4.7e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   0.78 | Tokens / Sec:   398.6 | Learning Rate: 4.6e-04
[mps] Epoch 9 Validation ====
(tensor(1.6499, device='mps:0'), <__main__.TrainState object at 0x123462840>)
[mps] Epoch 10 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   0.64 | Tokens / Sec:   545.4 | Learning Rate: 4.6e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   0.65 | Tokens / Sec:   595.9 | Learning Rate: 4.6e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   0.62 | Tokens / Sec:   585.5 | Learning Rate: 4.6e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   0.84 | Tokens / Sec:   583.9 | Learning Rate: 4.6e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   0.76 | Tokens / Sec:   587.1 | Learning Rate: 4.6e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   0.65 | Tokens / Sec:   587.8 | Learning Rate: 4.6e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   0.70 | Tokens / Sec:   583.6 | Learning Rate: 4.6e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   0.73 | Tokens / Sec:   580.9 | Learning Rate: 4.6e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   0.74 | Tokens / Sec:   580.4 | Learning Rate: 4.6e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   0.80 | Tokens / Sec:   587.9 | Learning Rate: 4.6e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   0.67 | Tokens / Sec:   586.9 | Learning Rate: 4.5e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   0.56 | Tokens / Sec:   587.0 | Learning Rate: 4.5e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   0.66 | Tokens / Sec:   582.9 | Learning Rate: 4.5e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   0.71 | Tokens / Sec:   584.8 | Learning Rate: 4.5e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   0.80 | Tokens / Sec:   585.3 | Learning Rate: 4.5e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   0.76 | Tokens / Sec:   591.0 | Learning Rate: 4.5e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   0.78 | Tokens / Sec:   589.5 | Learning Rate: 4.5e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   0.78 | Tokens / Sec:   582.4 | Learning Rate: 4.5e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   0.82 | Tokens / Sec:   585.1 | Learning Rate: 4.5e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   0.67 | Tokens / Sec:   590.1 | Learning Rate: 4.5e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   0.70 | Tokens / Sec:   590.6 | Learning Rate: 4.4e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   0.64 | Tokens / Sec:   589.1 | Learning Rate: 4.4e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   0.75 | Tokens / Sec:   599.7 | Learning Rate: 4.4e-04
[mps] Epoch 10 Validation ====
(tensor(1.6694, device='mps:0'), <__main__.TrainState object at 0x123462840>)
[mps] Epoch 11 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   0.59 | Tokens / Sec:   731.5 | Learning Rate: 4.4e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   0.66 | Tokens / Sec:   602.3 | Learning Rate: 4.4e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   0.58 | Tokens / Sec:   583.8 | Learning Rate: 4.4e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   0.51 | Tokens / Sec:   582.1 | Learning Rate: 4.4e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   0.61 | Tokens / Sec:   580.9 | Learning Rate: 4.4e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   0.59 | Tokens / Sec:   582.9 | Learning Rate: 4.4e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   0.72 | Tokens / Sec:   583.8 | Learning Rate: 4.4e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   0.60 | Tokens / Sec:   588.3 | Learning Rate: 4.4e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   0.59 | Tokens / Sec:   584.4 | Learning Rate: 4.4e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   0.68 | Tokens / Sec:   580.7 | Learning Rate: 4.3e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   0.61 | Tokens / Sec:   597.1 | Learning Rate: 4.3e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   0.60 | Tokens / Sec:   587.9 | Learning Rate: 4.3e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   0.74 | Tokens / Sec:   580.0 | Learning Rate: 4.3e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   0.69 | Tokens / Sec:   583.7 | Learning Rate: 4.3e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   0.63 | Tokens / Sec:   579.6 | Learning Rate: 4.3e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   0.69 | Tokens / Sec:   580.4 | Learning Rate: 4.3e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   0.69 | Tokens / Sec:   588.8 | Learning Rate: 4.3e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   0.61 | Tokens / Sec:   601.7 | Learning Rate: 4.3e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   0.63 | Tokens / Sec:   582.6 | Learning Rate: 4.3e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   0.59 | Tokens / Sec:   584.4 | Learning Rate: 4.3e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   0.69 | Tokens / Sec:   586.1 | Learning Rate: 4.3e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   0.52 | Tokens / Sec:   581.5 | Learning Rate: 4.2e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   0.61 | Tokens / Sec:   586.1 | Learning Rate: 4.2e-04
[mps] Epoch 11 Validation ====
(tensor(1.6673, device='mps:0'), <__main__.TrainState object at 0x123462840>)
[mps] Epoch 12 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   0.60 | Tokens / Sec:   676.3 | Learning Rate: 4.2e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   0.56 | Tokens / Sec:   607.3 | Learning Rate: 4.2e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   0.52 | Tokens / Sec:   580.1 | Learning Rate: 4.2e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   0.50 | Tokens / Sec:   589.4 | Learning Rate: 4.2e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   0.60 | Tokens / Sec:   590.4 | Learning Rate: 4.2e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   0.58 | Tokens / Sec:   594.5 | Learning Rate: 4.2e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   0.54 | Tokens / Sec:   592.1 | Learning Rate: 4.2e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   0.51 | Tokens / Sec:   582.4 | Learning Rate: 4.2e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   0.65 | Tokens / Sec:   548.8 | Learning Rate: 4.2e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   0.54 | Tokens / Sec:   563.4 | Learning Rate: 4.2e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   0.59 | Tokens / Sec:   567.5 | Learning Rate: 4.2e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   0.57 | Tokens / Sec:   575.5 | Learning Rate: 4.2e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   0.48 | Tokens / Sec:   576.6 | Learning Rate: 4.1e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   0.56 | Tokens / Sec:   583.9 | Learning Rate: 4.1e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   0.55 | Tokens / Sec:   570.0 | Learning Rate: 4.1e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   0.54 | Tokens / Sec:   569.0 | Learning Rate: 4.1e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   0.47 | Tokens / Sec:   571.5 | Learning Rate: 4.1e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   0.51 | Tokens / Sec:   568.8 | Learning Rate: 4.1e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   0.55 | Tokens / Sec:   570.0 | Learning Rate: 4.1e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   0.67 | Tokens / Sec:   581.1 | Learning Rate: 4.1e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   0.63 | Tokens / Sec:   582.2 | Learning Rate: 4.1e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   0.57 | Tokens / Sec:   581.3 | Learning Rate: 4.1e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   0.51 | Tokens / Sec:   573.2 | Learning Rate: 4.1e-04
[mps] Epoch 12 Validation ====
(tensor(1.6742, device='mps:0'), <__main__.TrainState object at 0x123462840>)
[mps] Epoch 13 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   0.46 | Tokens / Sec:   712.3 | Learning Rate: 4.1e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   0.52 | Tokens / Sec:   601.2 | Learning Rate: 4.1e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   0.57 | Tokens / Sec:   578.2 | Learning Rate: 4.1e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   0.50 | Tokens / Sec:   574.1 | Learning Rate: 4.0e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   0.52 | Tokens / Sec:   586.0 | Learning Rate: 4.0e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   0.46 | Tokens / Sec:   566.4 | Learning Rate: 4.0e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   0.43 | Tokens / Sec:   574.3 | Learning Rate: 4.0e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   0.45 | Tokens / Sec:   566.3 | Learning Rate: 4.0e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   0.42 | Tokens / Sec:   568.1 | Learning Rate: 4.0e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   0.44 | Tokens / Sec:   569.6 | Learning Rate: 4.0e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   0.55 | Tokens / Sec:   575.2 | Learning Rate: 4.0e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   0.43 | Tokens / Sec:   564.2 | Learning Rate: 4.0e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   0.49 | Tokens / Sec:   567.7 | Learning Rate: 4.0e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   0.50 | Tokens / Sec:   575.3 | Learning Rate: 4.0e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   0.49 | Tokens / Sec:   574.1 | Learning Rate: 4.0e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   0.56 | Tokens / Sec:   571.3 | Learning Rate: 4.0e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   0.52 | Tokens / Sec:   572.1 | Learning Rate: 4.0e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   0.49 | Tokens / Sec:   570.5 | Learning Rate: 4.0e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   0.45 | Tokens / Sec:   571.0 | Learning Rate: 4.0e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   0.48 | Tokens / Sec:   568.9 | Learning Rate: 3.9e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   0.57 | Tokens / Sec:   564.9 | Learning Rate: 3.9e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   0.53 | Tokens / Sec:   575.9 | Learning Rate: 3.9e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   0.50 | Tokens / Sec:   570.0 | Learning Rate: 3.9e-04
[mps] Epoch 13 Validation ====
(tensor(1.7271, device='mps:0'), <__main__.TrainState object at 0x123462840>)
[mps] Epoch 14 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   0.43 | Tokens / Sec:   749.5 | Learning Rate: 3.9e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   0.43 | Tokens / Sec:   592.5 | Learning Rate: 3.9e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   0.47 | Tokens / Sec:   566.1 | Learning Rate: 3.9e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   0.47 | Tokens / Sec:   578.4 | Learning Rate: 3.9e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   0.46 | Tokens / Sec:   563.1 | Learning Rate: 3.9e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   0.42 | Tokens / Sec:   574.5 | Learning Rate: 3.9e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   0.41 | Tokens / Sec:   574.2 | Learning Rate: 3.9e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   0.42 | Tokens / Sec:   566.4 | Learning Rate: 3.9e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   0.47 | Tokens / Sec:   568.3 | Learning Rate: 3.9e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   0.48 | Tokens / Sec:   573.4 | Learning Rate: 3.9e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   0.43 | Tokens / Sec:   560.2 | Learning Rate: 3.9e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   0.40 | Tokens / Sec:   567.4 | Learning Rate: 3.9e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   0.45 | Tokens / Sec:   578.5 | Learning Rate: 3.8e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   0.41 | Tokens / Sec:   569.1 | Learning Rate: 3.8e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   0.55 | Tokens / Sec:   568.5 | Learning Rate: 3.8e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   0.46 | Tokens / Sec:   566.9 | Learning Rate: 3.8e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   0.48 | Tokens / Sec:   568.5 | Learning Rate: 3.8e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   0.47 | Tokens / Sec:   568.0 | Learning Rate: 3.8e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   0.49 | Tokens / Sec:   572.1 | Learning Rate: 3.8e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   0.58 | Tokens / Sec:   574.4 | Learning Rate: 3.8e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   0.54 | Tokens / Sec:   568.4 | Learning Rate: 3.8e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   0.47 | Tokens / Sec:   568.5 | Learning Rate: 3.8e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   0.52 | Tokens / Sec:   565.7 | Learning Rate: 3.8e-04
[mps] Epoch 14 Validation ====
(tensor(1.7419, device='mps:0'), <__main__.TrainState object at 0x123462840>)
[mps] Epoch 15 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   0.39 | Tokens / Sec:   681.7 | Learning Rate: 3.8e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   0.37 | Tokens / Sec:   594.4 | Learning Rate: 3.8e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   0.34 | Tokens / Sec:   573.8 | Learning Rate: 3.8e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   0.43 | Tokens / Sec:   568.8 | Learning Rate: 3.8e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   0.42 | Tokens / Sec:   560.5 | Learning Rate: 3.8e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   0.42 | Tokens / Sec:   567.1 | Learning Rate: 3.8e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   0.43 | Tokens / Sec:   577.3 | Learning Rate: 3.8e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   0.43 | Tokens / Sec:   565.4 | Learning Rate: 3.8e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   0.37 | Tokens / Sec:   576.8 | Learning Rate: 3.7e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   0.37 | Tokens / Sec:   571.7 | Learning Rate: 3.7e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   0.44 | Tokens / Sec:   568.0 | Learning Rate: 3.7e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   0.37 | Tokens / Sec:   577.8 | Learning Rate: 3.7e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   0.46 | Tokens / Sec:   575.8 | Learning Rate: 3.7e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   0.41 | Tokens / Sec:   576.8 | Learning Rate: 3.7e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   0.41 | Tokens / Sec:   566.4 | Learning Rate: 3.7e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   0.42 | Tokens / Sec:   573.8 | Learning Rate: 3.7e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   0.46 | Tokens / Sec:   571.2 | Learning Rate: 3.7e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   0.37 | Tokens / Sec:   578.2 | Learning Rate: 3.7e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   0.46 | Tokens / Sec:   570.5 | Learning Rate: 3.7e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   0.44 | Tokens / Sec:   584.1 | Learning Rate: 3.7e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   0.38 | Tokens / Sec:   572.2 | Learning Rate: 3.7e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   0.44 | Tokens / Sec:   565.7 | Learning Rate: 3.7e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   0.46 | Tokens / Sec:   578.7 | Learning Rate: 3.7e-04
[mps] Epoch 15 Validation ====
(tensor(1.7471, device='mps:0'), <__main__.TrainState object at 0x123462840>)
[mps] Epoch 16 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   0.36 | Tokens / Sec:   737.6 | Learning Rate: 3.7e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   0.38 | Tokens / Sec:   591.1 | Learning Rate: 3.7e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   0.38 | Tokens / Sec:   576.6 | Learning Rate: 3.7e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   0.43 | Tokens / Sec:   573.2 | Learning Rate: 3.7e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   0.33 | Tokens / Sec:   568.1 | Learning Rate: 3.6e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   0.39 | Tokens / Sec:   570.4 | Learning Rate: 3.6e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   0.36 | Tokens / Sec:   558.8 | Learning Rate: 3.6e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   0.35 | Tokens / Sec:   577.7 | Learning Rate: 3.6e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   0.37 | Tokens / Sec:   584.9 | Learning Rate: 3.6e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   0.37 | Tokens / Sec:   571.6 | Learning Rate: 3.6e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   0.42 | Tokens / Sec:   573.6 | Learning Rate: 3.6e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   0.32 | Tokens / Sec:   572.9 | Learning Rate: 3.6e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   0.37 | Tokens / Sec:   568.3 | Learning Rate: 3.6e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   0.46 | Tokens / Sec:   581.4 | Learning Rate: 3.6e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   0.35 | Tokens / Sec:   562.5 | Learning Rate: 3.6e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   0.44 | Tokens / Sec:   572.2 | Learning Rate: 3.6e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   0.45 | Tokens / Sec:   571.3 | Learning Rate: 3.6e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   0.40 | Tokens / Sec:   574.4 | Learning Rate: 3.6e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   0.36 | Tokens / Sec:   562.2 | Learning Rate: 3.6e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   0.34 | Tokens / Sec:   575.1 | Learning Rate: 3.6e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   0.40 | Tokens / Sec:   562.8 | Learning Rate: 3.6e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   0.40 | Tokens / Sec:   563.1 | Learning Rate: 3.6e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   0.36 | Tokens / Sec:   573.6 | Learning Rate: 3.6e-04
[mps] Epoch 16 Validation ====
(tensor(1.7699, device='mps:0'), <__main__.TrainState object at 0x123462840>)
[mps] Epoch 17 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   0.30 | Tokens / Sec:   714.1 | Learning Rate: 3.6e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   0.30 | Tokens / Sec:   582.4 | Learning Rate: 3.6e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   0.38 | Tokens / Sec:   560.2 | Learning Rate: 3.5e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   0.30 | Tokens / Sec:   564.8 | Learning Rate: 3.5e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   0.32 | Tokens / Sec:   565.4 | Learning Rate: 3.5e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   0.33 | Tokens / Sec:   570.4 | Learning Rate: 3.5e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   0.31 | Tokens / Sec:   577.4 | Learning Rate: 3.5e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   0.38 | Tokens / Sec:   568.3 | Learning Rate: 3.5e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   0.33 | Tokens / Sec:   558.8 | Learning Rate: 3.5e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   0.35 | Tokens / Sec:   576.8 | Learning Rate: 3.5e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   0.36 | Tokens / Sec:   574.0 | Learning Rate: 3.5e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   0.33 | Tokens / Sec:   559.6 | Learning Rate: 3.5e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   0.33 | Tokens / Sec:   574.2 | Learning Rate: 3.5e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   0.39 | Tokens / Sec:   571.9 | Learning Rate: 3.5e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   0.41 | Tokens / Sec:   576.1 | Learning Rate: 3.5e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   0.41 | Tokens / Sec:   562.2 | Learning Rate: 3.5e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   0.38 | Tokens / Sec:   564.2 | Learning Rate: 3.5e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   0.33 | Tokens / Sec:   561.9 | Learning Rate: 3.5e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   0.34 | Tokens / Sec:   575.3 | Learning Rate: 3.5e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   0.36 | Tokens / Sec:   566.9 | Learning Rate: 3.5e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   0.38 | Tokens / Sec:   578.6 | Learning Rate: 3.5e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   0.30 | Tokens / Sec:   570.8 | Learning Rate: 3.5e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   0.40 | Tokens / Sec:   577.7 | Learning Rate: 3.5e-04
[mps] Epoch 17 Validation ====
(tensor(1.7791, device='mps:0'), <__main__.TrainState object at 0x123462840>)
[mps] Epoch 18 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   0.31 | Tokens / Sec:   680.4 | Learning Rate: 3.5e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   0.29 | Tokens / Sec:   581.9 | Learning Rate: 3.5e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   0.26 | Tokens / Sec:   570.1 | Learning Rate: 3.5e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   0.34 | Tokens / Sec:   571.2 | Learning Rate: 3.4e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   0.34 | Tokens / Sec:   567.8 | Learning Rate: 3.4e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   0.29 | Tokens / Sec:   566.6 | Learning Rate: 3.4e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   0.35 | Tokens / Sec:   571.1 | Learning Rate: 3.4e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   0.34 | Tokens / Sec:   569.0 | Learning Rate: 3.4e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   0.27 | Tokens / Sec:   572.3 | Learning Rate: 3.4e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   0.31 | Tokens / Sec:   560.8 | Learning Rate: 3.4e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   0.31 | Tokens / Sec:   574.5 | Learning Rate: 3.4e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   0.27 | Tokens / Sec:   568.5 | Learning Rate: 3.4e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   0.35 | Tokens / Sec:   562.9 | Learning Rate: 3.4e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   0.33 | Tokens / Sec:   573.5 | Learning Rate: 3.4e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   0.31 | Tokens / Sec:   578.1 | Learning Rate: 3.4e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   0.34 | Tokens / Sec:   575.3 | Learning Rate: 3.4e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   0.30 | Tokens / Sec:   567.7 | Learning Rate: 3.4e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   0.41 | Tokens / Sec:   567.7 | Learning Rate: 3.4e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   0.36 | Tokens / Sec:   571.5 | Learning Rate: 3.4e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   0.31 | Tokens / Sec:   574.6 | Learning Rate: 3.4e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   0.37 | Tokens / Sec:   568.2 | Learning Rate: 3.4e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   0.32 | Tokens / Sec:   572.9 | Learning Rate: 3.4e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   0.28 | Tokens / Sec:   563.1 | Learning Rate: 3.4e-04
[mps] Epoch 18 Validation ====
(tensor(1.8121, device='mps:0'), <__main__.TrainState object at 0x123462840>)
[mps] Epoch 19 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   0.32 | Tokens / Sec:   677.6 | Learning Rate: 3.4e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   0.27 | Tokens / Sec:   569.6 | Learning Rate: 3.4e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   0.28 | Tokens / Sec:   576.0 | Learning Rate: 3.4e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   0.29 | Tokens / Sec:   564.0 | Learning Rate: 3.4e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   0.31 | Tokens / Sec:   559.6 | Learning Rate: 3.4e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   0.30 | Tokens / Sec:   560.5 | Learning Rate: 3.3e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   0.31 | Tokens / Sec:   574.2 | Learning Rate: 3.3e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   0.28 | Tokens / Sec:   564.3 | Learning Rate: 3.3e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   0.31 | Tokens / Sec:   578.7 | Learning Rate: 3.3e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   0.33 | Tokens / Sec:   566.5 | Learning Rate: 3.3e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   0.31 | Tokens / Sec:   582.6 | Learning Rate: 3.3e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   0.25 | Tokens / Sec:   584.7 | Learning Rate: 3.3e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   0.32 | Tokens / Sec:   568.7 | Learning Rate: 3.3e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   0.29 | Tokens / Sec:   545.2 | Learning Rate: 3.3e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   0.37 | Tokens / Sec:   566.2 | Learning Rate: 3.3e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   0.34 | Tokens / Sec:   576.1 | Learning Rate: 3.3e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   0.27 | Tokens / Sec:   554.7 | Learning Rate: 3.3e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   0.30 | Tokens / Sec:   580.6 | Learning Rate: 3.3e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   0.32 | Tokens / Sec:   575.0 | Learning Rate: 3.3e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   0.32 | Tokens / Sec:   582.0 | Learning Rate: 3.3e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   0.30 | Tokens / Sec:   567.4 | Learning Rate: 3.3e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   0.32 | Tokens / Sec:   563.1 | Learning Rate: 3.3e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   0.31 | Tokens / Sec:   567.1 | Learning Rate: 3.3e-04
[mps] Epoch 19 Validation ====
(tensor(1.8250, device='mps:0'), <__main__.TrainState object at 0x123462840>)
[mps] Epoch 20 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   0.30 | Tokens / Sec:   636.0 | Learning Rate: 3.3e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   0.22 | Tokens / Sec:   586.1 | Learning Rate: 3.3e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   0.26 | Tokens / Sec:   957.0 | Learning Rate: 3.3e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   0.24 | Tokens / Sec:  1009.8 | Learning Rate: 3.3e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   0.25 | Tokens / Sec:  1039.5 | Learning Rate: 3.3e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   0.25 | Tokens / Sec:  1047.3 | Learning Rate: 3.3e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   0.24 | Tokens / Sec:  1046.4 | Learning Rate: 3.3e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   0.24 | Tokens / Sec:  1048.9 | Learning Rate: 3.3e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   0.27 | Tokens / Sec:  1056.8 | Learning Rate: 3.3e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   0.22 | Tokens / Sec:  1042.3 | Learning Rate: 3.2e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   0.30 | Tokens / Sec:  1047.8 | Learning Rate: 3.2e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   0.23 | Tokens / Sec:  1059.7 | Learning Rate: 3.2e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   0.29 | Tokens / Sec:  1050.0 | Learning Rate: 3.2e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   0.28 | Tokens / Sec:  1054.2 | Learning Rate: 3.2e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   0.27 | Tokens / Sec:  1059.2 | Learning Rate: 3.2e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   0.23 | Tokens / Sec:  1055.0 | Learning Rate: 3.2e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   0.30 | Tokens / Sec:  1066.7 | Learning Rate: 3.2e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   0.26 | Tokens / Sec:  1054.9 | Learning Rate: 3.2e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   0.31 | Tokens / Sec:  1045.4 | Learning Rate: 3.2e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   0.28 | Tokens / Sec:  1054.1 | Learning Rate: 3.2e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   0.26 | Tokens / Sec:  1050.1 | Learning Rate: 3.2e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   0.31 | Tokens / Sec:  1049.4 | Learning Rate: 3.2e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   0.28 | Tokens / Sec:  1051.6 | Learning Rate: 3.2e-04
[mps] Epoch 20 Validation ====
(tensor(1.8422, device='mps:0'), <__main__.TrainState object at 0x123462840>)
[mps] Epoch 21 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   0.23 | Tokens / Sec:  1334.5 | Learning Rate: 3.2e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   0.21 | Tokens / Sec:  1060.9 | Learning Rate: 3.2e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   0.26 | Tokens / Sec:  1067.0 | Learning Rate: 3.2e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   0.23 | Tokens / Sec:  1046.4 | Learning Rate: 3.2e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   0.24 | Tokens / Sec:  1040.3 | Learning Rate: 3.2e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   0.23 | Tokens / Sec:  1055.6 | Learning Rate: 3.2e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   0.20 | Tokens / Sec:  1048.1 | Learning Rate: 3.2e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   0.27 | Tokens / Sec:  1047.6 | Learning Rate: 3.2e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   0.27 | Tokens / Sec:  1038.5 | Learning Rate: 3.2e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   0.27 | Tokens / Sec:  1054.4 | Learning Rate: 3.2e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   0.25 | Tokens / Sec:  1057.4 | Learning Rate: 3.2e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   0.24 | Tokens / Sec:  1050.9 | Learning Rate: 3.2e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   0.30 | Tokens / Sec:  1047.1 | Learning Rate: 3.2e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   0.27 | Tokens / Sec:  1059.2 | Learning Rate: 3.2e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   0.31 | Tokens / Sec:  1058.1 | Learning Rate: 3.2e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   0.28 | Tokens / Sec:  1043.7 | Learning Rate: 3.2e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   0.32 | Tokens / Sec:  1058.9 | Learning Rate: 3.1e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   0.28 | Tokens / Sec:  1061.1 | Learning Rate: 3.1e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   0.28 | Tokens / Sec:  1045.8 | Learning Rate: 3.1e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   0.30 | Tokens / Sec:  1049.5 | Learning Rate: 3.1e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   0.26 | Tokens / Sec:  1037.7 | Learning Rate: 3.1e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   0.29 | Tokens / Sec:  1055.8 | Learning Rate: 3.1e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   0.29 | Tokens / Sec:  1068.5 | Learning Rate: 3.1e-04
[mps] Epoch 21 Validation ====
(tensor(1.8403, device='mps:0'), <__main__.TrainState object at 0x123462840>)
[mps] Epoch 22 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   0.23 | Tokens / Sec:  1344.2 | Learning Rate: 3.1e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   0.20 | Tokens / Sec:  1059.4 | Learning Rate: 3.1e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   0.20 | Tokens / Sec:  1049.4 | Learning Rate: 3.1e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   0.25 | Tokens / Sec:  1048.5 | Learning Rate: 3.1e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   0.25 | Tokens / Sec:  1044.4 | Learning Rate: 3.1e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   0.24 | Tokens / Sec:  1052.8 | Learning Rate: 3.1e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   0.20 | Tokens / Sec:  1082.0 | Learning Rate: 3.1e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   0.24 | Tokens / Sec:  1061.4 | Learning Rate: 3.1e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   0.19 | Tokens / Sec:  1067.2 | Learning Rate: 3.1e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   0.21 | Tokens / Sec:  1055.8 | Learning Rate: 3.1e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   0.22 | Tokens / Sec:  1062.8 | Learning Rate: 3.1e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   0.25 | Tokens / Sec:  1057.5 | Learning Rate: 3.1e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   0.26 | Tokens / Sec:  1041.5 | Learning Rate: 3.1e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   0.23 | Tokens / Sec:  1073.3 | Learning Rate: 3.1e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   0.22 | Tokens / Sec:  1072.1 | Learning Rate: 3.1e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   0.26 | Tokens / Sec:  1059.3 | Learning Rate: 3.1e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   0.20 | Tokens / Sec:  1037.3 | Learning Rate: 3.1e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   0.26 | Tokens / Sec:  1043.7 | Learning Rate: 3.1e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   0.25 | Tokens / Sec:  1066.2 | Learning Rate: 3.1e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   0.24 | Tokens / Sec:  1047.6 | Learning Rate: 3.1e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   0.27 | Tokens / Sec:  1072.8 | Learning Rate: 3.1e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   0.25 | Tokens / Sec:  1044.4 | Learning Rate: 3.1e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   0.25 | Tokens / Sec:  1052.4 | Learning Rate: 3.1e-04
[mps] Epoch 22 Validation ====
(tensor(1.8384, device='mps:0'), <__main__.TrainState object at 0x123462840>)
[mps] Epoch 23 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   0.21 | Tokens / Sec:  1348.6 | Learning Rate: 3.1e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   0.20 | Tokens / Sec:  1041.4 | Learning Rate: 3.1e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   0.20 | Tokens / Sec:  1073.4 | Learning Rate: 3.1e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   0.20 | Tokens / Sec:  1056.7 | Learning Rate: 3.1e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   0.21 | Tokens / Sec:  1075.9 | Learning Rate: 3.0e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   0.24 | Tokens / Sec:  1047.8 | Learning Rate: 3.0e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   0.19 | Tokens / Sec:  1054.5 | Learning Rate: 3.0e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   0.22 | Tokens / Sec:  1052.1 | Learning Rate: 3.0e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   0.21 | Tokens / Sec:  1057.7 | Learning Rate: 3.0e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   0.20 | Tokens / Sec:  1058.0 | Learning Rate: 3.0e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   0.18 | Tokens / Sec:  1054.5 | Learning Rate: 3.0e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   0.24 | Tokens / Sec:  1060.8 | Learning Rate: 3.0e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   0.21 | Tokens / Sec:  1035.7 | Learning Rate: 3.0e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   0.26 | Tokens / Sec:  1045.1 | Learning Rate: 3.0e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   0.23 | Tokens / Sec:  1057.2 | Learning Rate: 3.0e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   0.22 | Tokens / Sec:  1047.4 | Learning Rate: 3.0e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   0.24 | Tokens / Sec:  1047.0 | Learning Rate: 3.0e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   0.21 | Tokens / Sec:  1040.3 | Learning Rate: 3.0e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   0.24 | Tokens / Sec:  1045.2 | Learning Rate: 3.0e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   0.27 | Tokens / Sec:  1038.5 | Learning Rate: 3.0e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   0.27 | Tokens / Sec:  1040.5 | Learning Rate: 3.0e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   0.24 | Tokens / Sec:  1040.0 | Learning Rate: 3.0e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   0.21 | Tokens / Sec:  1051.7 | Learning Rate: 3.0e-04
[mps] Epoch 23 Validation ====
(tensor(1.8615, device='mps:0'), <__main__.TrainState object at 0x123462840>)
[mps] Epoch 24 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   0.22 | Tokens / Sec:  1381.2 | Learning Rate: 3.0e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   0.19 | Tokens / Sec:  1053.1 | Learning Rate: 3.0e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   0.23 | Tokens / Sec:  1041.4 | Learning Rate: 3.0e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   0.20 | Tokens / Sec:  1041.8 | Learning Rate: 3.0e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   0.21 | Tokens / Sec:  1036.9 | Learning Rate: 3.0e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   0.22 | Tokens / Sec:  1060.4 | Learning Rate: 3.0e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   0.23 | Tokens / Sec:  1058.8 | Learning Rate: 3.0e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   0.19 | Tokens / Sec:  1043.2 | Learning Rate: 3.0e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   0.19 | Tokens / Sec:  1060.5 | Learning Rate: 3.0e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   0.23 | Tokens / Sec:  1042.2 | Learning Rate: 3.0e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   0.23 | Tokens / Sec:  1059.6 | Learning Rate: 3.0e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   0.27 | Tokens / Sec:  1049.3 | Learning Rate: 3.0e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   0.20 | Tokens / Sec:  1047.1 | Learning Rate: 3.0e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   0.22 | Tokens / Sec:  1054.2 | Learning Rate: 3.0e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   0.21 | Tokens / Sec:  1044.6 | Learning Rate: 3.0e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   0.22 | Tokens / Sec:  1049.0 | Learning Rate: 3.0e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   0.22 | Tokens / Sec:  1058.1 | Learning Rate: 3.0e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   0.25 | Tokens / Sec:  1046.6 | Learning Rate: 2.9e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   0.21 | Tokens / Sec:  1040.6 | Learning Rate: 2.9e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   0.20 | Tokens / Sec:  1047.6 | Learning Rate: 2.9e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   0.19 | Tokens / Sec:  1041.9 | Learning Rate: 2.9e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   0.21 | Tokens / Sec:  1062.8 | Learning Rate: 2.9e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   0.22 | Tokens / Sec:  1048.5 | Learning Rate: 2.9e-04
[mps] Epoch 24 Validation ====
(tensor(1.8762, device='mps:0'), <__main__.TrainState object at 0x123462840>)
[mps] Epoch 25 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   0.22 | Tokens / Sec:  1296.7 | Learning Rate: 2.9e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   0.18 | Tokens / Sec:  1049.0 | Learning Rate: 2.9e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   0.17 | Tokens / Sec:  1044.1 | Learning Rate: 2.9e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   0.21 | Tokens / Sec:  1070.3 | Learning Rate: 2.9e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   0.17 | Tokens / Sec:  1054.0 | Learning Rate: 2.9e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   0.22 | Tokens / Sec:  1042.5 | Learning Rate: 2.9e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   0.23 | Tokens / Sec:  1047.5 | Learning Rate: 2.9e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   0.19 | Tokens / Sec:  1048.3 | Learning Rate: 2.9e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   0.21 | Tokens / Sec:  1045.9 | Learning Rate: 2.9e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   0.19 | Tokens / Sec:  1063.2 | Learning Rate: 2.9e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   0.21 | Tokens / Sec:  1052.4 | Learning Rate: 2.9e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   0.17 | Tokens / Sec:  1056.5 | Learning Rate: 2.9e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   0.20 | Tokens / Sec:  1062.8 | Learning Rate: 2.9e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   0.22 | Tokens / Sec:  1068.4 | Learning Rate: 2.9e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   0.24 | Tokens / Sec:  1059.1 | Learning Rate: 2.9e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   0.20 | Tokens / Sec:  1043.5 | Learning Rate: 2.9e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   0.21 | Tokens / Sec:  1063.9 | Learning Rate: 2.9e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   0.19 | Tokens / Sec:  1063.3 | Learning Rate: 2.9e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   0.21 | Tokens / Sec:  1051.9 | Learning Rate: 2.9e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   0.18 | Tokens / Sec:  1062.7 | Learning Rate: 2.9e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   0.23 | Tokens / Sec:  1050.3 | Learning Rate: 2.9e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   0.24 | Tokens / Sec:  1055.9 | Learning Rate: 2.9e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   0.19 | Tokens / Sec:  1055.5 | Learning Rate: 2.9e-04
[mps] Epoch 25 Validation ====
(tensor(1.8847, device='mps:0'), <__main__.TrainState object at 0x123462840>)
[mps] Epoch 26 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   0.21 | Tokens / Sec:  1283.3 | Learning Rate: 2.9e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   0.20 | Tokens / Sec:  1093.3 | Learning Rate: 2.9e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   0.16 | Tokens / Sec:  1081.7 | Learning Rate: 2.9e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   0.17 | Tokens / Sec:  1071.8 | Learning Rate: 2.9e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   0.19 | Tokens / Sec:  1059.1 | Learning Rate: 2.9e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   0.17 | Tokens / Sec:  1068.3 | Learning Rate: 2.9e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   0.16 | Tokens / Sec:  1075.0 | Learning Rate: 2.9e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   0.18 | Tokens / Sec:  1076.5 | Learning Rate: 2.9e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   0.23 | Tokens / Sec:  1069.1 | Learning Rate: 2.9e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   0.21 | Tokens / Sec:  1076.4 | Learning Rate: 2.9e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   0.19 | Tokens / Sec:  1073.7 | Learning Rate: 2.9e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   0.18 | Tokens / Sec:  1088.3 | Learning Rate: 2.9e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   0.18 | Tokens / Sec:  1075.8 | Learning Rate: 2.8e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   0.18 | Tokens / Sec:  1081.7 | Learning Rate: 2.8e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   0.17 | Tokens / Sec:  1072.5 | Learning Rate: 2.8e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   0.18 | Tokens / Sec:  1074.5 | Learning Rate: 2.8e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   0.18 | Tokens / Sec:  1056.5 | Learning Rate: 2.8e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   0.18 | Tokens / Sec:  1059.7 | Learning Rate: 2.8e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   0.22 | Tokens / Sec:  1075.5 | Learning Rate: 2.8e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   0.17 | Tokens / Sec:  1062.7 | Learning Rate: 2.8e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   0.21 | Tokens / Sec:  1068.1 | Learning Rate: 2.8e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   0.18 | Tokens / Sec:  1044.6 | Learning Rate: 2.8e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   0.20 | Tokens / Sec:  1067.9 | Learning Rate: 2.8e-04
[mps] Epoch 26 Validation ====
(tensor(1.9094, device='mps:0'), <__main__.TrainState object at 0x123462840>)
[mps] Epoch 27 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   0.17 | Tokens / Sec:  1369.3 | Learning Rate: 2.8e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   0.17 | Tokens / Sec:  1079.5 | Learning Rate: 2.8e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   0.18 | Tokens / Sec:  1076.2 | Learning Rate: 2.8e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   0.15 | Tokens / Sec:  1061.2 | Learning Rate: 2.8e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   0.17 | Tokens / Sec:  1069.1 | Learning Rate: 2.8e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   0.15 | Tokens / Sec:  1059.3 | Learning Rate: 2.8e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   0.16 | Tokens / Sec:  1071.8 | Learning Rate: 2.8e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   0.16 | Tokens / Sec:  1069.7 | Learning Rate: 2.8e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   0.21 | Tokens / Sec:  1052.7 | Learning Rate: 2.8e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   0.17 | Tokens / Sec:  1057.6 | Learning Rate: 2.8e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   0.18 | Tokens / Sec:  1067.0 | Learning Rate: 2.8e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   0.18 | Tokens / Sec:  1051.2 | Learning Rate: 2.8e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   0.17 | Tokens / Sec:  1062.8 | Learning Rate: 2.8e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   0.18 | Tokens / Sec:  1053.1 | Learning Rate: 2.8e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   0.16 | Tokens / Sec:  1065.8 | Learning Rate: 2.8e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   0.16 | Tokens / Sec:  1050.2 | Learning Rate: 2.8e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   0.19 | Tokens / Sec:  1063.3 | Learning Rate: 2.8e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   0.16 | Tokens / Sec:  1059.4 | Learning Rate: 2.8e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   0.19 | Tokens / Sec:  1062.1 | Learning Rate: 2.8e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   0.17 | Tokens / Sec:  1050.6 | Learning Rate: 2.8e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   0.21 | Tokens / Sec:  1040.7 | Learning Rate: 2.8e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   0.20 | Tokens / Sec:  1066.1 | Learning Rate: 2.8e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   0.20 | Tokens / Sec:  1048.6 | Learning Rate: 2.8e-04
[mps] Epoch 27 Validation ====
(tensor(1.8920, device='mps:0'), <__main__.TrainState object at 0x123462840>)
[mps] Epoch 28 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   0.18 | Tokens / Sec:  1354.8 | Learning Rate: 2.8e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   0.19 | Tokens / Sec:  1051.9 | Learning Rate: 2.8e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   0.17 | Tokens / Sec:  1041.1 | Learning Rate: 2.8e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   0.15 | Tokens / Sec:  1056.0 | Learning Rate: 2.8e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   0.15 | Tokens / Sec:  1064.0 | Learning Rate: 2.8e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   0.17 | Tokens / Sec:  1044.6 | Learning Rate: 2.8e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   0.17 | Tokens / Sec:  1054.5 | Learning Rate: 2.8e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   0.16 | Tokens / Sec:  1044.0 | Learning Rate: 2.8e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   0.17 | Tokens / Sec:  1070.6 | Learning Rate: 2.8e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   0.15 | Tokens / Sec:  1071.4 | Learning Rate: 2.8e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   0.19 | Tokens / Sec:  1060.9 | Learning Rate: 2.8e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   0.19 | Tokens / Sec:  1072.5 | Learning Rate: 2.7e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   0.18 | Tokens / Sec:  1068.3 | Learning Rate: 2.7e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   0.16 | Tokens / Sec:  1068.1 | Learning Rate: 2.7e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   0.13 | Tokens / Sec:  1072.0 | Learning Rate: 2.7e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   0.17 | Tokens / Sec:  1054.5 | Learning Rate: 2.7e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   0.20 | Tokens / Sec:  1067.5 | Learning Rate: 2.7e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   0.14 | Tokens / Sec:  1060.0 | Learning Rate: 2.7e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   0.19 | Tokens / Sec:  1061.1 | Learning Rate: 2.7e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   0.16 | Tokens / Sec:  1048.6 | Learning Rate: 2.7e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   0.16 | Tokens / Sec:  1073.3 | Learning Rate: 2.7e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   0.15 | Tokens / Sec:  1037.1 | Learning Rate: 2.7e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   0.16 | Tokens / Sec:  1056.7 | Learning Rate: 2.7e-04
[mps] Epoch 28 Validation ====
(tensor(1.9132, device='mps:0'), <__main__.TrainState object at 0x123462840>)
[mps] Epoch 29 Training ====
Epoch Step:      1 | Accumulation Step:   1 | Loss:   0.13 | Tokens / Sec:  1361.8 | Learning Rate: 2.7e-04
Epoch Step:     41 | Accumulation Step:   5 | Loss:   0.15 | Tokens / Sec:  1052.1 | Learning Rate: 2.7e-04
Epoch Step:     81 | Accumulation Step:   9 | Loss:   0.15 | Tokens / Sec:  1056.4 | Learning Rate: 2.7e-04
Epoch Step:    121 | Accumulation Step:  13 | Loss:   0.15 | Tokens / Sec:  1045.1 | Learning Rate: 2.7e-04
Epoch Step:    161 | Accumulation Step:  17 | Loss:   0.12 | Tokens / Sec:  1064.5 | Learning Rate: 2.7e-04
Epoch Step:    201 | Accumulation Step:  21 | Loss:   0.17 | Tokens / Sec:  1068.4 | Learning Rate: 2.7e-04
Epoch Step:    241 | Accumulation Step:  25 | Loss:   0.17 | Tokens / Sec:  1054.2 | Learning Rate: 2.7e-04
Epoch Step:    281 | Accumulation Step:  29 | Loss:   0.15 | Tokens / Sec:  1054.7 | Learning Rate: 2.7e-04
Epoch Step:    321 | Accumulation Step:  33 | Loss:   0.14 | Tokens / Sec:  1071.4 | Learning Rate: 2.7e-04
Epoch Step:    361 | Accumulation Step:  37 | Loss:   0.14 | Tokens / Sec:  1048.2 | Learning Rate: 2.7e-04
Epoch Step:    401 | Accumulation Step:  41 | Loss:   0.20 | Tokens / Sec:  1054.8 | Learning Rate: 2.7e-04
Epoch Step:    441 | Accumulation Step:  45 | Loss:   0.20 | Tokens / Sec:  1071.5 | Learning Rate: 2.7e-04
Epoch Step:    481 | Accumulation Step:  49 | Loss:   0.15 | Tokens / Sec:  1040.5 | Learning Rate: 2.7e-04
Epoch Step:    521 | Accumulation Step:  53 | Loss:   0.17 | Tokens / Sec:  1047.7 | Learning Rate: 2.7e-04
Epoch Step:    561 | Accumulation Step:  57 | Loss:   0.15 | Tokens / Sec:  1048.4 | Learning Rate: 2.7e-04
Epoch Step:    601 | Accumulation Step:  61 | Loss:   0.18 | Tokens / Sec:  1057.9 | Learning Rate: 2.7e-04
Epoch Step:    641 | Accumulation Step:  65 | Loss:   0.17 | Tokens / Sec:  1054.4 | Learning Rate: 2.7e-04
Epoch Step:    681 | Accumulation Step:  69 | Loss:   0.16 | Tokens / Sec:  1043.9 | Learning Rate: 2.7e-04
Epoch Step:    721 | Accumulation Step:  73 | Loss:   0.16 | Tokens / Sec:  1050.6 | Learning Rate: 2.7e-04
Epoch Step:    761 | Accumulation Step:  77 | Loss:   0.16 | Tokens / Sec:  1053.3 | Learning Rate: 2.7e-04
Epoch Step:    801 | Accumulation Step:  81 | Loss:   0.16 | Tokens / Sec:  1071.5 | Learning Rate: 2.7e-04
Epoch Step:    841 | Accumulation Step:  85 | Loss:   0.16 | Tokens / Sec:  1066.9 | Learning Rate: 2.7e-04
Epoch Step:    881 | Accumulation Step:  89 | Loss:   0.15 | Tokens / Sec:  1054.1 | Learning Rate: 2.7e-04
[mps] Epoch 29 Validation ====
(tensor(1.9105, device='mps:0'), <__main__.TrainState object at 0x123462840>)
```
---

### 🖥️ Inference & Decoding Visualizations

#### 10 Epochs Sample Output
![alt text](<assets/Screenshot 2026-08-30 at 11.09.43 AM.png>)

#### 20 Epochs Sample Output
![alt text](<assets/Screenshot 2026-08-30 at 11.08.44 AM.png>)

#### 30 Epochs Sample Output
![alt text](<assets/Screenshot 2026-08-30 at 11.09.43 AM.png>)

---

## 📜 Acknowledgments

Based on the paper *"Attention Is All You Need"* (Vaswani et al., 2017) and implementation references from [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/).
