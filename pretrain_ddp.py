import os
import socket
import json
import math

# -----------------------------
# MPI + Intel GPU / oneCCL imports
# -----------------------------
from mpi4py import MPI
import torch
import torch.distributed as dist
import oneccl_bindings_for_pytorch as torch_ccl
import intel_extension_for_pytorch as ipex
from torch.nn.parallel import DistributedDataParallel as DDP

# Standard PyTorch / Transformers imports
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

from tokenizers import ByteLevelBPETokenizer
from transformers import (
    PreTrainedTokenizerFast,
    PretrainedConfig,
    PreTrainedModel,
)
from tqdm import tqdm

#####################################
# 1) MPI-based Setup
#####################################

comm = MPI.COMM_WORLD
SIZE = comm.Get_size()    # total number of processes
RANK = comm.Get_rank()    # this process's rank
LOCAL_RANK = os.environ.get('PALS_LOCAL_RANKID', '0')

# Set environment variables used by PyTorch for DDP
os.environ['RANK'] = str(RANK)
os.environ['WORLD_SIZE'] = str(SIZE)

# Hostname logic for MASTER_ADDR
MASTER_ADDR = socket.gethostname() if RANK == 0 else None
MASTER_ADDR = comm.bcast(MASTER_ADDR, root=0)
os.environ['MASTER_ADDR'] = f"{MASTER_ADDR}.hsn.cm.aurora.alcf.anl.gov"
os.environ['MASTER_PORT'] = str(2345)

print(f"[DDP] Hi from rank {RANK} of {SIZE} with local rank {LOCAL_RANK}. MASTER_ADDR={os.environ['MASTER_ADDR']}")

# Initialize process group (oneCCL) for distributed comm
dist.init_process_group(
    backend='ccl',
    init_method='env://',
    rank=int(RANK),
    world_size=int(SIZE)
)

# Pin this process to the local XPU device
torch.xpu.set_device(int(LOCAL_RANK))
device = torch.device('xpu')
torch.manual_seed(0)

#####################################
# 2) Tokenizer Utilities
#####################################

def train_bpe_tokenizer(file_path, vocab_size=5000):
    """Train a ByteLevel BPE tokenizer on the text and save it in Hugging Face format."""
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train([file_path], vocab_size=vocab_size, min_frequency=2,
                    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
    os.makedirs("bpe_tokenizer", exist_ok=True)
    tokenizer.save_model("bpe_tokenizer")
    with open(os.path.join("bpe_tokenizer", "tokenizer.json"), "w", encoding="utf-8") as f:
        f.write(tokenizer._tokenizer.to_str())

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
# 3) Model Definition (GPT)
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

        # causal mask
        self.register_buffer("mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
                 .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x):
        B, T, C = x.size()
        q = self.query(x).view(B, T, self.n_head, self.head_dim).transpose(1,2)
        k = self.key(x).view(B, T, self.n_head, self.head_dim).transpose(1,2)
        v = self.value(x).view(B, T, self.n_head, self.head_dim).transpose(1,2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v
        y = y.transpose(1,2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, 4*config.n_embd)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4*config.n_embd, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

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

        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

        # This calls self.apply(self._init_weights)
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, "Sequence length exceeds block size"

        token_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding[:, :T, :]
        x = self.drop(token_emb + pos_emb)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=0.7, top_k=None):
        self.eval()
        generated = input_ids
        for _ in range(max_new_tokens):
            if generated.size(1) > self.config.block_size:
                idx_cond = generated[:, -self.config.block_size:]
            else:
                idx_cond = generated

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
        return generated

#####################################
# 4) GPT Training Loop
#####################################

def main():
    # Because we're now using MPI, we already have RANK, SIZE, device set above
    data_path = "data/Hongloumeng.txt"
    block_size = 256
    epochs = 500
    batch_size = 24
    lr = 3e-5

    # Rank 0: Train tokenizer if not present
    if not os.path.exists("bpe_tokenizer/vocab.json"):
        if RANK == 0:
            print("Training BPE tokenizer...")
            _ = train_bpe_tokenizer(data_path, vocab_size=5000)
        MPI.COMM_WORLD.barrier()

    # Load tokenizer, data
    hf_tokenizer = load_bpe_tokenizer()
    data, vocab_size = load_data_bpe(data_path, hf_tokenizer)
    dataset_size = len(data)
    print(f"[Rank {RANK}] dataset_size: {dataset_size}")
    batches_per_epoch = dataset_size // (batch_size * block_size)
    print(f"[Rank {RANK}] batches_per_epoch: {batches_per_epoch}")

    # Initialize model config
    config_model = GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.1
    )
    model = GPT(config_model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Optimize with IPEX before wrapping in DDP
    model, optimizer = ipex.optimize(model, optimizer=optimizer)

    # Wrap in DDP
    model = DDP(model)

    # AMP GradScaler
    scaler = torch.amp.GradScaler()
    global_step = 0

    # Training loop
    for epoch in tqdm(range(epochs), desc="Epochs"):
        for step in range(batches_per_epoch):
            global_step += 1
            x, y = get_batch(data, batch_size, block_size, device)
            optimizer.zero_grad()

            # AMP autocast on XPU
            with torch.amp.autocast(device_type='xpu', dtype=torch.bfloat16):
                _, loss = model(x, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Print / generate on rank 0
            if (global_step % 100 == 0) and (RANK == 0):
                print(f"[Rank 0] Epoch {epoch} Step {global_step} Loss {loss.item():.4f}")
                prompt_str = "只见这袭人在床上睡着了，"
                token_ids = hf_tokenizer.encode(prompt_str)
                prompt_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(device)
                # Generate text from model.module
                generated = model.module.generate(prompt_tensor, max_new_tokens=200)
                out_text = hf_tokenizer.decode(generated[0].cpu().tolist())
                print(f"\n--- Generated text @ step {global_step} ---\n{out_text}\n")

            # Checkpoint every 5000 steps
            if (global_step % 5000 == 0) and (RANK == 0):
                os.makedirs("pre-trained", exist_ok=True)
                ckpt = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "model_state_dict": model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss.item()
                }
                path = f"pre-trained/hongloumeng_checkpoint_step_{global_step}.pth"
                torch.save(ckpt, path)
                print(f"Checkpoint saved at step {global_step}")

    # Rank 0 saves final model + tokenizer
    if RANK == 0:
        model.module.save_pretrained("RedLLM")
        hf_tokenizer.save_pretrained("RedLLM")
        print("Training complete. Model + tokenizer saved in 'RedLLM'.")

    # Final cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    main()