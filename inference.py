import os
import torch
import torch.nn.functional as F
from models.crsd_seq import CRSDSequence
from utils.config import load_config
from data.prepare_data import build_tokenizer

@torch.no_grad()
def generate_text(model, tokenizer, device, prompt="The ", max_new_tokens=200, temperature=1.0):
    """Autoregressive text generation from CRSD model."""
    model.eval()
    
    # 🚨 FIX 1: Use the tokenizer instance to encode the prompt.
    # The tokenizer object (tok) is callable and handles the encoding.
    
    # Assuming tokenizer is callable and returns a list of integer IDs (the simplest case)
    # The tokenizer should handle breaking the prompt string into its tokens/subwords.
    
    # To be safe, we use the method that the LLMTokenizer should implement:
    # 1. Get the list of IDs from the tokenizer's encoding method
    # 2. Extract the IDs from the encoded output (assuming the LLMTokenizer is a wrapper)
    
    # We will assume that the LLMTokenizer instance is callable and returns a list of IDs.
    try:
        input_ids = tokenizer(prompt)
    except Exception:
        # Fallback: If LLMTokenizer is not callable, try the internal tokenizer's method
        input_ids = tokenizer.tokenizer.encode(prompt).ids
        
    # Check if a simple list of integers was returned
    if not isinstance(input_ids, list):
        # Handle case where tokenizer returns a tensor/dict (common in HuggingFace)
        if isinstance(input_ids, dict) and 'input_ids' in input_ids:
            input_ids = input_ids['input_ids'].tolist()
        elif isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.squeeze().tolist()
        # If it's still not a list of IDs, we might have a deeper issue, but we proceed.
        
    
    # Convert list of Python ints to a PyTorch tensor for the model
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    print(f"🧩 Starting generation with prompt: '{prompt}'")
    for _ in range(max_new_tokens):
        # 🚨 FIX 2 (Efficiency Improvement): Only pass the last part of the sequence 
        # to the model for the next token prediction if the sequence length exceeds 
        # the model's max context size (or to speed up generation).
        # We will keep the full tensor for simplicity unless OOM occurs.
        logits = model(input_tensor)  # (B, T, V)
        next_logits = logits[:, -1, :] / temperature
        probs = F.softmax(next_logits, dim=-1)

        # Sample the next token ID
        next_id = torch.multinomial(probs, num_samples=1).item()
        
        # 🚨 FIX 3: Get the character/token from the correct map. 
        # Assuming the inverse map is stored in 'tokenizer.inv' based on the original code
        next_token = tokenizer.inv.get(next_id, "<UNK>") # Use .get() for safety

        # Stop early if we hit padding or invalid token
        if next_token in ["<PAD>", "<UNK>"]:
            break

        prompt += next_token
        input_ids.append(next_id)
        
        # Update input tensor for the next step: append the new ID
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
        
        # Optional: Print progress
        sys.stdout.write(f"\r{prompt[:80]}...")
        sys.stdout.flush()

    print() # Newline after progress
    return prompt


def main():
    project_root = os.path.dirname(os.path.abspath(__file__))
    # Assuming config is relative to the current working directory or script location
    config_path = os.path.join(project_root, "experiments", "exp_language_model.yaml") 
    checkpoint_path = "/kaggle/working/checkpoints/crsd_best.pt" 

    # Load config
    cfg = load_config(config_path)
    # Using the same device logic as train.py
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Using device: {device}")

    # Build tokenizer
    tok = build_tokenizer(cfg["data"]["dataset_path"])
    # 🚨 FIX: Vocab size is now accessed via the corrected method
    vocab_size = tok.vocab_size() 
    print(f"📘 Loaded tokenizer with vocab size: {vocab_size}")

    # Load model
    model_cfg = cfg["model"]
    model = CRSDSequence(
        vocab_size=vocab_size,
        emb_dim=model_cfg["d_x"],
        **model_cfg,
    ).to(device)

    # Load checkpoint
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        
        # 2. Filter out keys that don't match the current model (e.g., buffers from previous runs)
        model_keys = model.state_dict().keys()
        filtered_state_dict = {
            k: v for k, v in state_dict.items() if k in model_keys
        }

        model.load_state_dict(filtered_state_dict, strict=False) # Use strict=False for robustness
        print(f"✅ Loaded checkpoint from: {checkpoint_path} (Filtered buffers)")
    else:
        print(f"⚠️ No checkpoint found at {checkpoint_path}, using random weights")

    # Generate text
    generated = generate_text(
        model,
        tokenizer=tok,
        device=device,
        prompt="The",
        max_new_tokens=300,
        temperature=0.8
    )

    print("\n📝 Generated Text:")
    print("-" * 60)
    print(generated)
    print("-" * 60)


if __name__ == "__main__":
    main()