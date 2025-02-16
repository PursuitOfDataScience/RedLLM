from tokenizers import ByteLevelBPETokenizer
import torch
from pretrain import GPT, load_bpe_tokenizer, GPTConfig

tokenizer = load_bpe_tokenizer()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Initialize your model
vocab_size = tokenizer.get_vocab_size()
block_size = 256  # context window size
config_model = GPTConfig(vocab_size=vocab_size, 
                            block_size=block_size,
                            n_layer=12, 
                            n_head=12, 
                            n_embd=768, 
                            dropout=0.1)

model = GPT(config_model)

# Load the pre-trained weights
checkpoint_path = "pre-trained/hongloumeng_final.pth"
state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=True)
model.load_state_dict(state_dict)

# Set the model to evaluation mode
model.eval()

model.to(device)

# Specify desired prompt string
prompt_str = "花袭人有始有终"
prompt_tokens = tokenizer.encode(prompt_str)
prompt_ids = prompt_tokens.ids  # Extract list of integers from the Encoding object
prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0).to(device)

# Generate text starting from the prompt.
generated = model.generate(prompt_tensor, max_new_tokens=400)
generated_text = tokenizer.decode(generated[0].tolist())
print(f"\n--- Generated text \n{generated_text}\n")