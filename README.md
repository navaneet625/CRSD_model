# CRSD Minimal Repo

Minimal implementation of the Cue-Revival State Dynamics (CRSD) prototype.

Run a tiny demo:

```bash
python train.py --config experiments/exp_crsd_tiny.yaml



🔍 Core Features

🧩 Modular Architecture – clean directory structure (models/, data/, utils/, scripts/) for reproducible experiments.

🔡 Tokenization System – supports character, word, and subword modes using custom tokenizers (Byte-level, Word, SentencePiece).

⚙️ Dataset Pipeline – robust dataloader for large-scale text corpora (e.g. enwik8_10M.txt), automatically splits, pads, and batches sequences.

📈 Training Loop – full training + evaluation pipeline with gradient scaling, checkpointing, and metrics (Loss, Accuracy, Bits-Per-Character, Perplexity).

🧠 Research Focus – designed for experimenting with modern SSM families: S4, S4D, S5, Mamba, LinOSS, Samba, and RetNet-inspired decoders.

🔬 Debug-Friendly – detailed print and inspection tools to trace tokenization, data flow, and model structure (detects silent data corruption, e.g. PAD flooding).


HebbianMemory(B, d_k, d_v)

Implements a differentiable associative memory that stores feature correlations 
in a dynamic matrix H ∈ ℝ^{B×d_k×d_v}. Each update reinforces associations 
between keys (k) and values (v) using Hebbian plasticity rules:

    H ← γH + η·(k⊗v)

At recall, given a query c, it retrieves the most relevant value using 
cosine-normalized similarity with optional top-k sparsification.

Features:
- Cosine-normalized recall for numerical stability
- Learnable decay (γ) and learning rate (η)
- Optional top-k selective recall
- Detach-safe memory (can disable gradient flow through memory)
- AMP / mixed-precision friendly


EpisodicBuffer(B, slots, d_k, d_v)

Implements a priority-based episodic memory that stores the most important 
(key, value) pairs over time, replacing the lowest-importance entries when full. 
It supports differentiable cosine-based retrieval, allowing global sequence 
context recall over recent history.

Features:
- Vectorized priority write (replace lowest-importance slot)
- Cosine similarity recall with temperature scaling
- Optional top-k or window-limited recall
- Stable normalization with eps guards
- Fully differentiable (detach_mem=False)
- Efficient for large batch or slot counts (AMP-compatible)


🧠 Theoretical Model Summary — CRSD: Contextual Recurrent Spectral Dual-Memory Model

The CRSD architecture (Contextual Recurrent Spectral Dual-memory) is a hybrid neural sequence model designed to combine the temporal sensitivity of recurrent networks, the global mixing efficiency of spectral transforms, and the adaptive recall capacity of biologically inspired memory systems.

At its core, each CRSD cell integrates three complementary computational pathways:

Recurrent Reservoir Dynamics —
A continuous-time update mechanism combines the current input, previous hidden state, and an internal “reservoir” representation through parameterized linear transformations.
This enables fine-grained temporal modeling with efficient local recurrence, preserving short-term dependencies.

Spectral Dual Transform (FFT-based A→B→A Mixing) —
Each hidden activation undergoes a LayerNorm → FFT → iFFT → Linear sub-block that transforms feature activations into the frequency domain, performs low-cost global mixing, and returns to the time domain.
This unitary (lossless) operation provides global context propagation across feature dimensions in O(d log d) complexity while maintaining gradient stability and energy preservation.

Dual-Memory Retrieval System —
CRSD maintains two interacting memory subsystems:

A Hebbian associative memory, which continuously integrates key–value correlations for long-term knowledge retention through the update rule
H ← γH + η(k⊗v).

An Episodic buffer, a slot-based short-term store that retains recent experiences based on importance scores and priority-based replacement.
Both memories are queried via normalized key vectors, producing context embeddings (v_hebb, v_buf) that are adaptively merged via a learnable gate into a unified recall vector (v̂_t).

The hidden state update is then gated by this composite recall signal, allowing each CRSD cell to dynamically blend local recurrence with global spectral mixing and contextual memory retrieval.

Formally, CRSD achieves a tri-domain integration:
time-domain recurrence (short-range),
frequency-domain transformation (global context),
and memory-domain retrieval (persistent knowledge).

This combination yields a model that scales efficiently to long sequences (O(T × d log d)), avoids gradient degradation through unitary spectral operations, and supports both online (recurrent) and offline (sequence-parallel) modes of processing.

In practice, CRSD exhibits transformer-level expressiveness with the efficiency and stability of RNNs, making it suitable for language modeling, continual learning, and dynamic sequence reasoning.