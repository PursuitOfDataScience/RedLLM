from transformers import AutoTokenizer
from mp_pretrain import GPTConfig, GPTModelParallel
import torch

# 1) Load model & tokenizer
loaded_config = GPTConfig.from_pretrained("RedLLM_MP")
loaded_tokenizer = AutoTokenizer.from_pretrained("RedLLM_MP")
loaded_model = GPTModelParallel.from_pretrained("RedLLM_MP", config=loaded_config)
loaded_model.eval()

# 2) Compute model parameters
total_params = sum(p.numel() for p in loaded_model.parameters())
print(f"Total number of parameters: {total_params}")

# 3) Prepare input on the same device
prompt = "花袭人有始有终，"
input_ids = loaded_tokenizer.encode(prompt, return_tensors="pt")

# 5) Generate (inference)
with torch.no_grad():
    output_ids = loaded_model.generate(
        input_ids,
        max_new_tokens=500,
        temperature=0.8,
        top_k=50
    )

# 6) Decode
print(loaded_tokenizer.decode(output_ids[0], skip_special_tokens=True))