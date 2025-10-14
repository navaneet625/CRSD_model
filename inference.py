import os,sys
import torch
import torch.nn.functional as F
from models.crsd_seq import CRSDSequence
from utils.config import load_config
from data.prepare_data import build_tokenizer

@torch.no_grad()
def generate_text(model, tokenizer, device, prompt="The ", max_new_tokens=200, temperature=1.0):
    """Autoregressive text generation from CRSD model."""
    model.eval()
    
    try:
        input_ids = tokenizer.tokenizer.encode(prompt) # 👈 REMOVED .ids
    except AttributeError:
        input_ids = tokenizer.tokenizer.encode(prompt)
        
    
    # Convert list of Python ints to a PyTorch tensor
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    print(f"🧩 Starting generation with prompt: '{prompt}'")
    for _ in range(max_new_tokens):
        # ... (rest of the generation logic is correct)
        
        logits = model(input_tensor)  # (B, T, V)
        next_logits = logits[:, -1, :] / temperature
        probs = F.softmax(next_logits, dim=-1)

        next_id = torch.multinomial(probs, num_samples=1).item()
        
        # 🚨 NOTE: 'tokenizer.inv' is the inverse map (ID to token)
        next_token = tokenizer.inv.get(next_id, "<UNK>") 

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
    vocab_size = tok.vocab_size() 

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
