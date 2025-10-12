crsd_model/
│
├── models/
│   ├── __init__.py
│   ├── crsd_cell.py           # Core: CRSDCell (reservoirs + hebbian + recall)
│   ├── crsd_block.py          # Multi-layer / stacked CRSDCell (like RNN/GRU stacking)
│   ├── crsd_seq.py            # Full sequence model (encoder / decoder or autoregressive)
│   └── crsd_memory.py         # Implements Hebbian + episodic memory modules separately
│
├── utils/
│   ├── __init__.py
│   ├── math_utils.py          # Small helpers (GELU, softmax variants, normalization)
│   ├── memory_utils.py        # Functions for updating/reading from memory
│   ├── visualization.py       # For plotting activations, memory attention maps
│   └── config.py              # YAML/JSON config loader for training
│
├── data/
│   ├── __init__.py
│   ├── dataloader.py          # Handles text datasets (e.g., PTB, WikiText, etc.)
│   ├── tokenizer.py           # Custom or HuggingFace tokenizer integration
│   └── prepare_data.py        # Data preprocessing / train-test split
│
├── experiments/
│   ├── exp_crsd_tiny.yaml     # Example config for toy “The quick brown fox” test
│   ├── exp_language_model.yaml# For large-scale LM training
│   ├── exp_ablation.yaml      # For testing components (reservoir vs memory vs gate)
│   └── logs/                  # TensorBoard / wandb logs
│
├── train.py                   # Training loop (supports multi-GPU, checkpointing)
├── eval.py                    # Evaluation + metrics
├── inference.py               # Text generation or recall testing
├── visualize_memory.py        # Visualize reservoir states and recall attention
├── run_pipeline.py            # End-to-end script to run training + evaluation
│
├── tests/
│   ├── test_crsd_math.py      # Unit tests for math correctness (like step_verbose)
│   ├── test_memory.py         # Test Hebbian + episodic recall functions
│   └── test_integration.py    # Full step-through on small sentence
│
├── docs/
│   ├── architecture.md        # Theory, math derivation, and explanation (like what we did)
│   ├── equations.md           # All equations (matrix form, symbolic form)
│   └── experiments.md         # Log of experimental results and ablations
│
└── README.md                  # Overview, model concept, and citation
# CRSD_model
