import os
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import intel_extension_for_pytorch as ipex  # IPEX import
from tqdm import tqdm
from tokenizers import ByteLevelBPETokenizer
from transformers import (
    PreTrainedTokenizerFast,
    PretrainedConfig,
    PreTrainedModel
)

import torch.distributed as dist
import oneccl_bindings_for_pytorch as torch_ccl  # so PyTorch can use the 'ccl' backend

# 1) Get rank/world_size from env vars set by mpiexec, e.g. RANK=0..(world_size-1).
rank = int(os.environ.get("RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))

# 2) Initialize the distributed process group with oneCCL.
dist.init_process_group(
    backend="ccl",
    init_method="env://",
    rank=rank,
    world_size=world_size
)

print(f"[Init] rank={rank}, world_size={world_size}")


#####################################
# BPE Tokenizer Utilities
#####################################

def train_bpe_tokenizer(file_path, vocab_size=5000):
    """Train a ByteLevel BPE tokenizer on text and save in Hugging Face format."""
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        [file_path],
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    )
    os.makedirs("bpe_tokenizer", exist_ok=True)
    tokenizer.save_model("bpe_tokenizer")

    # Save the tokenizer JSON
    with open(os.path.join("bpe_tokenizer", "tokenizer.json"), "w", encoding="utf-8") as f:
        f.write(tokenizer._tokenizer.to_str())

    # Create tokenizer_config.json
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

    # Create HF PreTrainedTokenizerFast
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
    return PreTrainedTokenizerFast.from_pretrained("bpe_tokenizer", use_fast=True)

def load_data_bpe(file_path, tokenizer):
    """Read and encode text with BPE. Return a tensor of token IDs."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    token_ids = tokenizer.encode(text)
    data = torch.tensor(token_ids, dtype=torch.long)
    return data, tokenizer.vocab_size

def get_batch(data, batch_size, block_size, device):
    """Sample a batch of contiguous sequences from data on a specified device."""
    n = data.size(0)
    ix = torch.randint(0, n - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]).to(device)
    return x, y

#####################################
# GPT Model Definition
#####################################

class GPTConfig(PretrainedConfig):
    model_type = "gpt_custom"
    def __init__(
        self,
        vocab_size=5000,
        block_size=1024,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout

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
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)

        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        # Causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
                 .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x):
        b, t, c = x.size()
        q = self.query(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = self.key(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = self.value(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :t, :t] == 0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(b, t, c)
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

#############################################
# GPT Model with Manual Pipeline Parallelism
#############################################

class GPTModelParallel(PreTrainedModel):
    """
    Manually pipeline-parallel GPT across Intel XPU devices in a single process:
      - We detect XPU devices via torch.xpu.device_count().
      - Split Transformer blocks across these devices.
      - Forward pass is from device 0 -> device 1 -> ... -> last device.
    """

    config_class = GPTConfig

    def __init__(self, config):
        super().__init__(config)

        # 1) Detect all XPU devices
        xpu_count = torch.xpu.device_count()
        if xpu_count == 0:
            raise ValueError("No XPU devices available (torch.xpu.device_count() == 0).")
        self.devices = [torch.device(f"xpu:{i}") for i in range(xpu_count)]

        # 2) Create embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.dropout)

        # 3) Build all the blocks on CPU initially
        all_blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        # Final LN + head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Initialize position embedding
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        self.post_init()

        # 4) Split blocks across XPUs
        num_xpus = len(self.devices)
        blocks_per_xpu = math.ceil(config.n_layer / num_xpus)
        self.pipeline_stages = nn.ModuleList()

        start_idx = 0
        for i in range(num_xpus):
            end_idx = min(start_idx + blocks_per_xpu, config.n_layer)
            stage_blocks = all_blocks[start_idx:end_idx]
            stage = nn.Sequential(*stage_blocks).to(self.devices[i])
            self.pipeline_stages.append(stage)
            start_idx = end_idx
            if end_idx >= config.n_layer:
                break

        # 5) Embeddings on device 0
        self.token_embedding.to(self.devices[0])
        self.position_embedding = nn.Parameter(self.position_embedding.to(self.devices[0]))
        self.drop.to(self.devices[0])

        # 6) Final LN + head on the last device
        self.ln_f.to(self.devices[-1])
        self.head.to(self.devices[-1])

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # Start on device[0]
        x = idx.to(self.devices[0])
        b, t = x.size()
        assert t <= self.config.block_size, f"Sequence length {t} > block_size {self.config.block_size}"

        # Embeddings on device[0]
        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding[:, :t, :]
        hidden_states = self.drop(token_emb + pos_emb)

        # Forward pass through pipeline stages
        for stage_idx, stage in enumerate(self.pipeline_stages):
            hidden_states = hidden_states.to(self.devices[stage_idx])
            hidden_states = stage(hidden_states)

        # On the last device, apply ln_f & head
        hidden_states = hidden_states.to(self.devices[-1])
        hidden_states = self.ln_f(hidden_states)
        logits = self.head(hidden_states)

        loss = None
        if targets is not None:
            targets = targets.to(self.devices[-1])
            logits = logits.view(-1, logits.size(-1))
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=0.7, top_k=None):
        self.eval()
        if len(self.devices) == 0:
            raise ValueError("No XPU devices found for generation.")

        # Start on device[0]
        generated = input_ids.to(self.devices[0])

        for _ in range(max_new_tokens):
            if generated.shape[1] > self.config.block_size:
                generated = generated[:, -self.config.block_size:]

            logits, _ = self.forward(generated)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                values, _ = torch.topk(logits, top_k)
                logits[logits < values[:, -1:]] = float('-inf')

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            next_token = next_token.to(self.devices[0])
            generated = torch.cat((generated, next_token), dim=1)

        return generated

#####################################
# Training Loop (Multi-Node + Multi-XPU)
#####################################

def train_xpu_model_parallel(data_path="data/Hongloumeng.txt"):
    rank = dist.get_rank()
    # 1) Prepare or load tokenizer
    if not os.path.exists("bpe_tokenizer/vocab.json"):
        print("Training BPE tokenizer...")
        train_bpe_tokenizer(data_path, vocab_size=5000)
    tokenizer = load_bpe_tokenizer()

    # 2) Load data
    data, vocab_size = load_data_bpe(data_path, tokenizer)
    block_size = 2048
    batch_size = 24
    epochs = 500  # Example shorter for demonstration

    # 3) Create config & model
    config_model = GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=24,
        n_head=24,
        n_embd=1296,
        dropout=0.1
    )
    base_model = GPTModelParallel(config_model)

    from torch.nn.parallel import DistributedDataParallel as DDP
    # Wrap the pipeline-parallel model in DDP
    model = DDP(base_model, device_ids=None)  # device_ids=None is correct for pipeline parallel

    # 4) Define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

    # (Optional) use ipex.optimize for performance
    model, optimizer = ipex.optimize(model, optimizer=optimizer)

    # 5) AMP GradScaler for XPU
    scaler = torch.amp.GradScaler()

    # Basic training settings
    dataset_size = len(data)
    batches_per_epoch = dataset_size // (batch_size * block_size)
    print(f"Dataset size: {dataset_size}, Batches/epoch: {batches_per_epoch}")

    global_step = 0
    for epoch in tqdm(range(epochs)):
        for _ in range(batches_per_epoch):
            global_step += 1

            # Because model is now DDP-wrapped, the underlying pipeline model is model.module
            first_device = model.module.devices[0]

            x, y = get_batch(data, batch_size, block_size, device=first_device)

            optimizer.zero_grad()
            # autocast for XPU
            with torch.amp.autocast(device_type='xpu'):
                logits, loss = model(x, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if global_step % 10 == 0 and rank == 0:
                print(f"Epoch {epoch} | Step {global_step} | Loss: {loss.item():.4f}")
                # quick generation example
                prompt_str = "只见这袭人在床上睡着了，"
                token_ids = tokenizer.encode(prompt_str)
                prompt_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)

                # We must call generate() on the underlying module
                generated = model.module.generate(prompt_tensor, max_new_tokens=20)
                out_txt = tokenizer.decode(generated[0].tolist())
                print(f"--- Generated text @ step {global_step} ---\n{out_txt}\n")

    # 6) Save final model & tokenizer
    # Save from the underlying module, not the DDP wrapper
    model.module.save_pretrained("RedLLM_MP_XPU")
    tokenizer.save_pretrained("RedLLM_MP_XPU")

    print("Multi-node, multi-XPU pipeline-parallel + DDP training complete. Model saved in RedLLM_MP_XPU.")

def main():
    if hasattr(torch, "xpu"):
        xpu_count = torch.xpu.device_count()
        print("XPU device count (via torch.xpu):", xpu_count)
    else:
        print("torch.xpu is not available.")
    
    train_xpu_model_parallel()

if __name__ == "__main__":
    main()