# CRSD Minimal Repository

**Minimal implementation of the Cue-Revival State Dynamics (CRSD) prototype.**

---

## 🚀 Demo: Quick Start

Clone the repo and run a tiny experiment:

```bash
git clone https://github.com/navaneet625/CRSD_model.git
cd CRSD_model
python train.py --config experiments/exp_crsd_tiny.yaml
```

---

## 🔍 Core Features

- **🧩 Modular Architecture**  
  Clean directory structure (`models/`, `data/`, `utils/`, `scripts/`), facilitating reproducible experiments.

- **🔡 Tokenization System**  
  Supports character, word, and subword modes using custom tokenizers:
  - Byte-level
  - Word
  - SentencePiece

- **⚙️ Dataset Pipeline**  
  Robust dataloader handles large-scale text corpora (e.g., `enwik8_10M.txt`).  
  Supports automatic data splitting, padding, and batching.

- **📈 Training Loop**  
  Complete training & evaluation pipeline:
  - Gradient scaling
  - Checkpointing
  - Metrics: Loss, Accuracy, Bits-Per-Character, Perplexity

- **🧠 Research Focus**  
  Designed for rapid experimentation with modern State Space Model (SSM) families:
  - S4, S4D, S5, Mamba, LinOSS, Samba
  - RetNet-inspired decoders

- **🔬 Debug-Friendly**  
  Inspection tools for tracing tokenization, data flow, model structure
  - Detects silent data corruption (e.g., PAD flooding)
  - Detailed print/debug modes

---
## 🏛️ Dual Memory Subsystems

### HebbianMemory

A differentiable associative memory storing feature correlations in a dynamic matrix `H` of shape `(B, d_k, d_v)`.

**Update rule:**  
```
H <- gamma * H + eta * outer(k, v)
```
Where:
- `gamma` is a learnable decay parameter  
- `eta` is a learnable learning rate  
- `outer(k, v)` denotes the outer product between key `k` and value `v`

**Features:**
- Cosine-normalized recall for stability
- Learnable decay (`gamma`) and learning rate (`eta`)
- Optional top-k selective recall
- Detach-safe memory (can disable gradient flow)
- Mixed-precision / AMP-friendly

---

### EpisodicBuffer

A prioritized, slot-based episodic memory. Retains important `(key, value)` pairs and replaces least important when full.

**Features:**
- Vectorized priority writing (replace lowest-importance slot)
- Cosine similarity recall with temperature scaling
- Optional top-k or window-limited recall
- Stable normalization (with epsilon for numerical safety)
- Fully differentiable
- Efficient for large batch or slot counts (AMP compatible)
---

## 🧠 Model Summary: Contextual Recurrent Spectral Dual-Memory (CRSD)

**CRSD = Contextual Recurrent Spectral Dual-Memory**

A hybrid neural sequence model combining:

1. **Recurrent Reservoir Dynamics**  
   Continuous-time, parameterized updates mixing current input, previous hidden state, and an internal reservoir for fine-grained temporal modeling (efficient local recurrence).

2. **Spectral Dual Transform**  
   Each hidden state passes through LayerNorm → FFT → iFFT → Linear, enabling efficient, global mixing in the frequency domain:
   - O(d log d) complexity
   - Gradient- and energy-preserving (unitary)

3. **Dual-Memory Retrieval System**  
   Two complementary memory modules:
   - **Hebbian associative memory** for long-term key–value correlations:  
     $$ H \leftarrow \gamma H + \eta (k \otimes v) $$
   - **Episodic buffer** for recent/priority-based experience recall
   - Both are queried and adaptively merged via a learnable gate for unified recall

**Tri-domain Integration:**  
- Time-domain recurrence (short-range)
- Frequency-domain transform (global context)
- Memory-domain retrieval (persistent knowledge)

_CRSD merges transformer-level expressivity with RNN-like efficiency and stability; suitable for language modeling, continual learning, and dynamic sequence reasoning._

---

## 📁 Directory Structure

```
CRSD_model/
│
├── models/        # Model architectures (CRSDCell, memory modules, etc.)
├── data/          # Data loaders, preprocessing, tokenization
├── utils/         # Helper functions, training utilities
├── scripts/       # Experiment scripts and entrypoints
├── experiments/   # Experiment configs (YAML)
├── train.py       # Training/evaluation loop
└── README.md
```

---

## References
- [Original S4 (Structured State Space Models)](https://arxiv.org/abs/2111.00396)
- [Recurrent Memory Transformers](https://arxiv.org/abs/2206.07162)
- [Frequency Domain Sequence Models](https://arxiv.org/abs/2301.12348)
- [Mamba: Linear-time SSMs](https://github.com/state-spaces/mamba)
- [RetNet: Retentive Networks](https://arxiv.org/abs/2307.08621)

---

## Contact

For questions, suggestions, or collaborations:
- GitHub: [navaneet625](https://github.com/navaneet625)
- Issues & discussions: [GitHub Issues](https://github.com/navaneet625/CRSD_model/issues)

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
