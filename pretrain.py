import os
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from tokenizers import ByteLevelBPETokenizer
from transformers import (
    PreTrainedTokenizerFast,
    PretrainedConfig,
    PreTrainedModel,
    AutoModelForCausalLM,
    AutoTokenizer
)

#####################################
# BPE Tokenizer Utilities
#####################################

def train_bpe_tokenizer(file_path, vocab_size=5000):
    """Train a ByteLevel BPE tokenizer on the text and save it in Hugging Face format."""
    # Train using the tokenizers library
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train([file_path], vocab_size=vocab_size, min_frequency=2,
                    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
    os.makedirs("bpe_tokenizer", exist_ok=True)
    # Save raw vocab and merges files (for reference)
    tokenizer.save_model("bpe_tokenizer")
    # Save the full tokenizer JSON representation
    with open(os.path.join("bpe_tokenizer", "tokenizer.json"), "w", encoding="utf-8") as f:
        f.write(tokenizer._tokenizer.to_str())
    # Create a tokenizer configuration
    tokenizer_config = {
        "model_max_length": 1024,
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "mask_token": "<mask>"
    }
    with open(os.path.join("bpe_tokenizer", "tokenizer_config.json"), "w") as f:
        json.dump(tokenizer_config, f)
    # Create a Hugging Face PreTrainedTokenizerFast instance using the saved tokenizer.json file
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join("bpe_tokenizer", "tokenizer.json"),
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>"
    )
    hf_tokenizer.save_pretrained("bpe_tokenizer")
    return hf_tokenizer

def load_bpe_tokenizer():
    """Load a previously trained BPE tokenizer in Hugging Face format."""
    hf_tokenizer = PreTrainedTokenizerFast.from_pretrained("bpe_tokenizer", use_fast=True)
    return hf_tokenizer

def load_data_bpe(file_path, tokenizer):
    """Read the text file, encode it with the BPE tokenizer, and return a tensor of token IDs."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    # Use the returned list of token IDs directly
    token_ids = tokenizer.encode(text)
    data = torch.tensor(token_ids, dtype=torch.long)
    return data, tokenizer.vocab_size

def get_batch(data, batch_size, block_size, device):
    """Randomly sample a batch of contiguous sequences from the tokenized data."""
    n = data.size(0)
    ix = torch.randint(0, n - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]).to(device)
    return x, y

#####################################
# Model Definition - Updated for HF compatibility
#####################################

class GPTConfig(PretrainedConfig):
    model_type = "gpt_custom"
    def __init__(self, vocab_size=5000, block_size=1024, n_layer=12, n_head=12, n_embd=768, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout

class GPT(PreTrainedModel):
    config_class = GPTConfig
    def __init__(self, config):
        super().__init__(config)
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Manually initialize parameters not handled by modules (e.g. position_embedding)
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        # post_init will call self.apply(self._init_weights)
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.config.block_size, "Sequence length exceeds block size"
        token_embeddings = self.token_embedding(idx)
        position_embeddings = self.position_embedding[:, :t, :]
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, input_ids, max_new_tokens, temperature=0.7, top_k=None):
        self.eval()
        generated = input_ids
        for _ in range(max_new_tokens):
            input_cond = generated if generated.shape[1] <= self.config.block_size else generated[:, -self.config.block_size:]
            logits, _ = self(input_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, top_k)
                logits[logits < values[:, -1:]] = -float('Inf')
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
        return generated

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "Embedding dim must be divisible by n_head"
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
    def forward(self, x):
        b, t, c = x.size()
        q = self.query(x).view(b, t, self.n_head, self.head_dim).transpose(1,2)
        k = self.key(x).view(b, t, self.n_head, self.head_dim).transpose(1,2)
        v = self.value(x).view(b, t, self.n_head, self.head_dim).transpose(1,2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :t, :t] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1,2).contiguous().view(b, t, c)
        y = self.resid_drop(self.proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

#####################################
# Distributed Training Setup for DDP
#####################################

# def setup(rank, world_size):
#     dist.init_process_group("nccl", rank=rank, world_size=world_size)

def setup(rank, world_size):
    # Explicitly set the CUDA device for the current process
    torch.cuda.set_device(rank)
    # Initialize the distributed process group using the NCCL backend
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size
    )

def cleanup():
    dist.destroy_process_group()

def train_ddp(rank, world_size, data_path="data/Hongloumeng.txt"):
    setup(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    
    # Only rank 0 trains the tokenizer if needed.
    if not os.path.exists("bpe_tokenizer/vocab.json"):
        if rank == 0:
            print("Training BPE tokenizer...")
            _ = train_bpe_tokenizer(data_path, vocab_size=5000)
        dist.barrier()
    hf_tokenizer = load_bpe_tokenizer()
    
    # Load data using the tokenizer
    data, vocab_size = load_data_bpe(data_path, hf_tokenizer)
    block_size = 256
    epochs = 500

    config_model = GPTConfig(vocab_size=vocab_size, 
                             block_size=block_size,
                             n_layer=12, 
                             n_head=12, 
                             n_embd=768, 
                             dropout=0.1)
    
    model = GPT(config_model).to(device)
    model = DDP(model, device_ids=[rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)


    # adding AMP autoscaler
    scaler = torch.amp.GradScaler("cuda")

    batch_size = 24

    dataset_size = len(data)
    print('dataset_size:', dataset_size)
    batches_per_epoch = dataset_size // (batch_size * block_size)
    print('batches_per_epoch:', batches_per_epoch)
    global_step = 0

    # from transformers import get_linear_schedule_with_warmup

    # num_training_steps = epochs * batches_per_epoch
    # num_warmup_steps = int(0.1 * num_training_steps)

    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer, 
    #     num_warmup_steps=num_warmup_steps, 
    #     num_training_steps=num_training_steps
    # )

    for epoch in tqdm(range(epochs)):
        for batch in range(batches_per_epoch): 
            global_step += 1
            x, y = get_batch(data, batch_size, block_size, device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                logits, loss = model(x, y)  # Forward pass is now automatically cast
            scaler.scale(loss).backward()      # Scale loss and call backward
            scaler.step(optimizer)             # Unscale gradients, check for infs/NaNs, and update parameters
            scaler.update()                    # Update the scaler for the next iteration

            if global_step % 100 == 0 and rank == 0:
                print(f"Rank {rank} | Epoch {epoch} | Step {global_step} | Loss: {loss.item():.4f}")
                prompt_str = "只见这袭人在床上睡着了，"
                token_ids = hf_tokenizer.encode(prompt_str)
                prompt_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(device)
                generated = model.module.generate(prompt_tensor, max_new_tokens=200)
                generated_text = hf_tokenizer.decode(generated[0].tolist())
                print(f"\n--- Generated text at step {global_step} ---\n{generated_text}\n")

            if global_step % 5000 == 0 and rank == 0:
                checkpoint = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss.item()
                }
                torch.save(checkpoint, f"pre-trained/hongloumeng_checkpoint_step_{global_step}.pth")
                print(f"Checkpoint saved at step {global_step}")

        # scheduler.step() # Update the learning rate at the end of each epoch
    
    if rank == 0:
        # Save the final model and tokenizer in Hugging Face format
        model.module.save_pretrained("RedLLM")
        hf_tokenizer.save_pretrained("RedLLM")
        print("DDP training complete; model and tokenizer saved in 'RedLLM'")
    cleanup()

def main():
    world_size = torch.cuda.device_count()
    mp.spawn(train_ddp, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
