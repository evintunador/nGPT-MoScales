import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import time
import copy
import glob
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import itertools
import tiktoken
import json
import math

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
torch.empty(1, device="cuda", requires_grad=True).backward() # prevents a bug on some systems
torch.set_float32_matmul_precision('high')
from torch import Tensor, nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.attention.flex_attention import BlockMask, flex_attention, create_block_mask

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the model

from kernels.cos_norm import cosine_norm_triton, cosine_norm_triton_

class Scale(nn.Module):
    """
    A module that manages learnable scaling parameters to ensure different learning rates
    from the rest of the parameters in the model (see pages 5 and 19)
    
    Args:
        dim (int): Dimension of the scaling parameter
        scale (float): Initial scale value
        init (float): Initial value for the scaling parameter
        device (str, optional): Device to store the parameter on
    """
    def __init__(self, dim: int, heads: int = 1, scale: float = 1.0, init: float = 1.0):
        super().__init__()
        self.init = init
        self.scale = scale
        self.s = nn.Parameter(torch.ones(heads, dim) * scale)
            # heads == 1 gives us a single regular vector
            # heads > 1 gets used in attention mechanism for different scaling vector for each head
    
    def forward(self):
        """Compute the effective scaling factor."""
        return self.s * (self.init / self.scale) # shape (heads, dim)

class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        # Ensure we don't exceed the dimension size
        dim_quarter = max(1, dim // 4)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim_quarter, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim_quarter)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq) # outer product
        self.cos = nn.Buffer(theta.cos(), persistent=False)
        self.sin = nn.Buffer(theta.sin(), persistent=False)

    def forward(self, x_BTHD: Tensor):
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = self.cos[None, :x_BTHD.size(-3), None, :], self.sin[None, :x_BTHD.size(-3), None, :]
        # Handle case where the number of dimensions is smaller
        dim_half = x_BTHD.size(-1) // 2
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos[..., :dim_half] + x2 * sin[..., :dim_half]
        y2 = x1 * (-sin[..., :dim_half]) + x2 * cos[..., :dim_half]
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, head_dim=None):
        super().__init__()
        # Calculate head_dim based on model dimensions and num_heads
        self.num_heads = num_heads
        # If head_dim not specified, calculate it based on the model dimension
        self.head_dim = dim // num_heads if head_dim is None else head_dim
        self.Wq = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.Wk = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.Wv = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        # the scaling factor to apply to the normalized queries & keys (see page 4)
        self.s_qk = Scale(self.head_dim, heads=num_heads, scale = 1. / math.sqrt(dim))
        # the scaling factor to apply to the attention logits to restore a variance of 1 (see page 4)
        self.scale = self.head_dim ** 0.5
        self.rotary = Rotary(self.head_dim, max_seq_len)
        self.Wo = nn.Linear(num_heads * self.head_dim, dim, bias=False)

    def forward(self, x: Tensor, block_mask: BlockMask):
        B, T = x.size(0), x.size(1) # batch size, sequence length
        assert B == 1, "Must use batch size = 1 for FlexAttention"
        # Linear projections for queries, keys, and values
        q, k, v = self.Wq(x), self.Wk(x), self.Wv(x)
            # shape: (batch_size, seq_len, dim) -> (batch_size, seq_len, num_heads * head_dim)
        # Reshape projections to separate heads
        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.num_heads, self.head_dim)
        v = v.view(B, T, self.num_heads, self.head_dim)
        # normalizing & scaling our queries  & keys (see page 4)
        s_qk = self.s_qk() # (num_heads, head_dim)
        q = cosine_norm_triton(q) * s_qk # then scale each head
        k = cosine_norm_triton(k) * s_qk # no shape change
        # apply RoPE
        q, k = self.rotary(q), self.rotary(k)
        # the meat of the attention calculation
        y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=block_mask, scale=0.12).transpose(1, 2)
        # combine heads
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim) # re-assemble all head outputs side by side
        # mix heads
        y = self.Wo(y)
        return y

class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: int = 4):
        super().__init__()
        hdim = int(mlp_ratio * dim)
        self.Wup = nn.Linear(dim, hdim, bias=False)
        self.Wgate = nn.Linear(dim, hdim, bias=False)
        self.Wdown = nn.Linear(hdim, dim, bias=False)
        # this flag designates Wdown to have a different parameter initialization as defined in model.py
        self.Wdown.GPT_scale_init = 1 
        # the learnable scaling factors
        self.s_u = Scale(hdim)
        self.s_v = Scale(hdim)
        # the varaince-controlling scaling term, needed to benefit from SiLU (see appendix A.1)
        self.scale = math.sqrt(dim)

    def forward(self, x: Tensor):
        # our up & gate projections
        u = self.Wup(x) # (batch_size, seq_len, hidden_dim)
        v = self.Wgate(x)
        # scale them
        u = u * self.s_u()
        v = v * self.s_v() * self.scale 
        # now perform the nonlinearity gate
        hidden = u * F.silu(v) # (batch_size, seq_len, hidden_dim)
        return self.Wdown(hidden) # (batch_size, seq_len, output_dim)

class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int, max_seq_len: int, layer_idx: int):
        super().__init__()
        self.attn = CausalSelfAttention(dim, num_heads, max_seq_len)
        self.mlp = MLP(dim, mlp_ratio)

        self.alpha_A = Scale(dim, init = 0.05, scale = 1. / math.sqrt(dim))
            # not sure what scale to use with a_A and a_M. At one point i had it as 1./math.sqrt(cfg.dim)
            # but now i can't find the reference to that in the paper
        # eigen learning rate vector
        self.alpha_M = Scale(dim, init = 0.05, scale = 1. / math.sqrt(dim))

    def forward(self, x: Tensor, block_mask: BlockMask):
        x_A = cosine_norm_triton(self.attn(x, block_mask))
        x = cosine_norm_triton(x + self.alpha_A() * (x_A - x))
        x_M = cosine_norm_triton(self.mlp(x))
        x = cosine_norm_triton(x + self.alpha_M() * (x_M - x))
        return x

# -----------------------------------------------------------------------------
# The main model

def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, model_dim: int, max_seq_len: int, mlp_ratio: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.model_dim = model_dim
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.blocks = nn.ModuleList([Block(model_dim, num_heads, mlp_ratio, max_seq_len, i) for i in range(num_layers)])
        # there are only 50257 unique GPT-2 tokens; we extend to nearest multiple of 128 for efficiency. this originates from Karpathy's experiments.
        self.lm_head = nn.Linear(model_dim, next_multiple_of_n(vocab_size, n=128))
        # scaling param to un-limit the range for the final probability distribution (see page 2)
        self.s_z = Scale(next_multiple_of_n(vocab_size, n=128), scale = 1./math.sqrt(model_dim))
        # initializing params to specific distributions
        self.apply(self.__init__weights)

    def __init__weights(self, module):
        """
        parameter initialization isn't actually important in N-GPT because of the normalization
        However we'll still initialize according to how they did in appendix A.5
        """
        # whereas GPT-2 used std = 0.02, we'll do square root of model's embedding dimension
        std = math.sqrt(self.model_dim) 

        if isinstance(module, (nn.Linear, nn.Parameter)):
            # specific weight matrices at the end of each layer are given smaller std 
            # originally this was done in GPT-2 to keep the residual stream small
            if hasattr(module, 'GPT_scale_init'):
                std *= (2 * len(self.blocks)) ** -0.5

            # carries out the actual initialization
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

            # biases, if any, should instead be initialized to zeros
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias) 

        # the embedding matrix doesn't count as an nn.Linear so we've gotta do it again for that
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, input_seq: Tensor, target_seq: Tensor = None):
        assert input_seq.ndim == 1

        docs = (input_seq == 50256).cumsum(0)
        def document_causal(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            document_mask = docs[q_idx] == docs[kv_idx]
            return causal_mask & document_mask
        doc_causal_mask = create_block_mask(document_causal, B=None, H=None, Q_LEN=input_seq.size(0), KV_LEN=input_seq.size(0))
      
        x = self.embed(input_seq)[None]

        for i in range(len(self.blocks)):
            x = self.blocks[i](x, doc_causal_mask)

        logits = self.lm_head(x).float()
        # to un-limit the temperature of the final probability distribution (see page 2)
        scaled_logits = logits * self.s_z()
        
        if target_seq is None:
            return scaled_logits
        else:
            return F.cross_entropy(scaled_logits.view(-1, logits.size(-1)), target_seq, 
                                  reduction='sum' if self.training else 'mean')

    def get_num_params(self):
        """
        Return the number of parameters in the model.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params
    
    def normalize_linear(self, module):
        """
        Helper method to normalize Linear layer weights where one dimension matches model dim
        """
        # Find the dimension that matches cfg.dim
        dim_to_normalize = None
        for dim, size in enumerate(module.weight.shape):
            if size == self.model_dim:
                dim_to_normalize = dim
                break
        
        if dim_to_normalize is not None:
            # Normalize the weights in-place
            cosine_norm_triton_(module.weight.data, dim=dim_to_normalize)

    def enforce_constraints(self):
        """
        Enforces constraints after each optimization step:
        1. Absolute value constraint on eigen learning rate parameters
        2. Cosine normalization on Linear layer weights where one dimension matches model dim
        """
        with torch.no_grad():
            # Enforce absolute value on eigen learning rates
            for layer in self.blocks:
                layer.alpha_A.s.data.abs_()
                layer.alpha_M.s.data.abs_()
            
            # Cosine normalize relevant Linear layers
            for module in self.modules():
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    self.normalize_linear(module)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        assert idx.ndim == 1
        def cdiv(m, n):
            return (m + (n - 1)) // n
        seq_len = idx.size(0)
        if seq_len % 128 != 0:
            pad_ct = cdiv(seq_len, 128) * 128 - seq_len
            idx = torch.cat((idx, torch.zeros(pad_ct, dtype=idx.dtype, device=idx.device)), dim=0)
        
        self.eval()  # Ensure model is in evaluation mode
        for _ in range(max_new_tokens):
            # Forward pass to get logits
            logits = self(idx[-self.max_seq_len:] if idx.size(0) > self.max_seq_len else idx)
            # Focus on the last token's prediction
            logits = logits[0, min(seq_len, self.max_seq_len) - 1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[-1]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            while idx_next >= self.vocab_size: # don't want to grab any of the out-of-vocab tokens
                idx_next = torch.multinomial(probs, num_samples=1)
                # could get stuck in an infinite loop here bit that's hella unlikely & i'm lazy
            # append sampled index to the running sequence and continue
            idx[min(seq_len, self.max_seq_len)] = idx_next

            # iterate sequence count and account for any time we surpass flex-attention's block size
            seq_len += 1
            if (seq_len - 1) % 128 == 0:
                pad_ct = cdiv(seq_len, 128) * 128 - seq_len
                idx = torch.cat((idx, [0] * pad_ct), dim=0)

        return idx[:seq_len]

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32) # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2]) # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True) # avoid pin_memory copy by @YouJiacheng
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy()) # avoid bytes->array copy by @YouJiacheng
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

def distributed_data_generator(filename_pattern: str, batch_size: int, rank: int, world_size: int):
    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    if not files:
        raise ValueError(f"No files found matching pattern: {filename_pattern}")
    
    assert batch_size % world_size == 0
    local_batch_size = batch_size // world_size
    
    # Calculate total tokens across all shards
    total_tokens = 0
    tokens_per_file = []
    for file in files:
        header = torch.from_file(str(file), False, 256, dtype=torch.int32)
        file_tokens = int(header[2])
        total_tokens += file_tokens
        tokens_per_file.append(file_tokens)
    
    # Calculate how many tokens we need for training
    tokens_needed = args.num_iterations * batch_size
    
    # Determine if we need to cycle and calculate epochs
    will_cycle = total_tokens < tokens_needed
    epochs = tokens_needed / total_tokens if total_tokens > 0 else 0
    
    if rank == 0:
        print0(f"Total tokens across {len(files)} shard(s): {total_tokens:,}", console=True)
        print0(f"Tokens needed for {args.num_iterations} iterations: {tokens_needed:,}", console=True)
        print0(f"Training will use approximately {epochs:.2f} epochs over the data", console=True)
    
    file_iter = itertools.cycle(files) if will_cycle else iter(files)
    tokens, pos = _load_data_shard(next(file_iter)), 0
    
    while True:
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        buf = tokens[pos + rank * local_batch_size:][:local_batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True) # no sync on host side;
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True) # H2D in another stream isn't helpful.
        pos += batch_size
        yield inputs, targets

# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    kernels = True
    # data
    train_files = "data/fineweb*10B/fineweb*_train_*.bin" # input .bin to train on
    val_files = "data/fineweb*10B/fineweb*_val_*.bin" # input .bin to eval validation loss on
    val_tokens = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons
    train_seq_len = 8*1024 # FlexAttention sequence length - reduced from 48*1024 for GPUs w/ at least 8GB VRAM during testing
    val_seq_len = 8*1024 # FlexAttention sequence length for validation - reduced from 4*64*1024
    # optimization
    num_iterations = 200 # number of iterations to run
    lr_init = 0.001
    lr_final = 0.0001
    # architecture
    vocab_size = 50257
    # model size - setup for GPUs w/ 8GB of VRAM
    num_layers = 6
    num_heads = 6
    model_dim = 384
    head_dim = None  # if None, will be set to model_dim // num_heads
    mlp_ratio = int(4 * 2/3) # 2/3 to make the GLU number of parameters eqiuvalent to not GLU
    # evaluation and logging
    val_loss_every = 100 # every how many steps to evaluate val loss? 0 for only at the end
    save_checkpoint = False

    def __post_init__(self):
        # Validate and set derived parameters
        if self.head_dim is None:
            self.head_dim = self.model_dim // self.num_heads
        assert self.head_dim in [2 ** i for i in range(1, 10)], f"head_dim must be a power of 2, got {self.head_dim}"
        assert self.mlp_ratio > 0, f"mlp_ratio must be positive, got {self.mlp_ratio}"
        assert self.train_seq_len % 128 == 0, f"train_seq_len must be multiple of 128, got {self.train_seq_len}"
        assert self.val_seq_len % 128 == 0, f"val_seq_len must be multiple of 128, got {self.val_seq_len}"

args = Hyperparameters()

# torchrun sets these env variables
rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
# Remove assertion for 8xH100
# assert world_size == 8 # this code is designed for 8xH100
print(f"Running with {world_size} GPUs (designed originally for 8xH100, adapted to also support 2x GPUs w/ at least 8GB VRAM during testing)")
assert torch.cuda.is_available()
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group(backend="nccl", device_id=device)
dist.barrier()
master_process = (rank == 0) # this process will do logging, checkpointing etc.

# begin logging
logfile = None
if master_process:
    run_id = uuid.uuid4()
    os.makedirs("experiments", exist_ok=True)  # Changed from "logs" to "experiments"
    logfile = f"experiments/{run_id}.txt"  # Changed from "logs/" to "experiments/"
    print(logfile)
def print0(s, console=False):
    if master_process:
        with open(logfile, "a") as f:
            if console:
                print(s)
            print(s, file=f)

# begin by printing all relevant files
print0(code)  # Print this file's code

# Print hellaswag.py if it exists
try:
    with open("hellaswag.py", "r") as f:
        print0("\n" + "="*100 + "\nhellaswag.py:\n" + "="*100)
        print0(f.read())
except FileNotFoundError:
    print0("\n" + "="*100 + "\nhellaswag.py not found\n" + "="*100)

print0("="*100)

# log information about the hardware/software environment this is running on
print0(f"Running Python {sys.version}")
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")
def nvidia_smi():
    import subprocess  # avoid top level import
    return subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True).stdout
print0(nvidia_smi())
print0("="*100)

########################################
#    Construct model and optimizer     #
########################################

model: nn.Module = GPT(vocab_size=args.vocab_size, 
                       num_layers=args.num_layers,
                       num_heads=args.num_heads, 
                       model_dim=args.model_dim,
                       max_seq_len=max(args.train_seq_len, args.val_seq_len),
                       mlp_ratio=args.mlp_ratio).cuda()
print0(f'{model.get_num_params()} parameters', console=True)
print0(model)

for param in model.parameters():
    dist.broadcast(param.detach(), 0)

# Simple optimizer initialization - zero weight decay
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_init, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)

# Learning rate schedule without warmup
def get_lr(step: int): # TODO add warmup for regular GPT
    # Cosine decay phase only
    cosine_decay = 0.5 * (1 + math.cos(math.pi * step / args.num_iterations))
    return max(cosine_decay, args.lr_final / args.lr_init)

# Use a more memory-efficient compilation option
model: nn.Module = torch.compile(model, dynamic=False, mode="reduce-overhead")

# Add fallback mode to handle compilation errors
import torch._dynamo
torch._dynamo.config.suppress_errors = True

########################################
#            Warmup kernels            #
########################################

# Attempt to limit memory fragmentation
if hasattr(torch.cuda, 'memory_stats'):
    print0(f"Initial GPU memory: {torch.cuda.memory_allocated() // (1024 * 1024)} MB")

# Warmup the training kernels, then re-initialize the state so we aren't cheating
warmup_steps = 10
initial_state = dict(model=copy.deepcopy(model.state_dict()),
                     optimizer=copy.deepcopy(optimizer.state_dict())) # save the initial state
for _ in range(warmup_steps):
    inputs = targets = torch.randint(0, args.vocab_size, size=(args.train_seq_len,), device="cuda")
    model(inputs.to(torch.int32), targets).backward()
    for param in model.parameters():
        dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
    optimizer.step()
    model.zero_grad(set_to_none=True)
model.load_state_dict(initial_state["model"])
optimizer.load_state_dict(initial_state["optimizer"])
del initial_state

if hasattr(torch.cuda, 'memory_stats'):
    print0(f"After warmup GPU memory: {torch.cuda.memory_allocated() // (1024 * 1024)} MB")

########################################
#        Training and validation       #
########################################

def sample_from_model(model, prompt, max_new_tokens=100, temperature=0.8, top_k=200):
    """Generate text samples from the model given a prompt."""
    # We need an encoding function - assuming you'll use tiktoken or similar
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)
    
    # Encode the prompt
    input_ids = encode(prompt)
    x = torch.tensor(input_ids, dtype=torch.int32, device="cuda")
    
    # Generate
    model.eval()
    with torch.no_grad():
        y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
    
    # Decode and return
    return decode(y.tolist())

train_loader = distributed_data_generator(args.train_files, world_size * args.train_seq_len, rank, world_size)

training_time_ms = 0
# start the clock
torch.cuda.synchronize()
t0 = time.perf_counter()
# begin training
train_steps = args.num_iterations
for step in range(train_steps + 1):
    last_step = (step == train_steps)

    # --------------- VALIDATION SECTION -----------------
    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
        # stop the clock
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.perf_counter() - t0)
        model.eval()
        
        # Use smaller val batch for GPUs w/ at least 8GB VRAM during testing
        val_batch_size = world_size * args.val_seq_len
        # Ensure we validate on enough tokens while keeping memory usage reasonable
        val_steps = max(1, min(16, args.val_tokens // val_batch_size))
        val_tokens_used = val_batch_size * val_steps
        
        print0(f"Validating on {val_tokens_used} tokens ({val_steps} steps with {val_batch_size} batch size)")
        
        # Choose between real data loader and synthetic data loader for validation
        val_loader = distributed_data_generator(args.val_files, val_batch_size, rank, world_size)
        val_loss = 0
        with torch.no_grad():
            for i in range(val_steps):
                inputs, targets = next(val_loader)
                # Check if inputs exceed sequence length
                if inputs.size(0) > args.val_seq_len:
                    inputs = inputs[:args.val_seq_len]
                    targets = targets[:args.val_seq_len]
                val_loss += model(inputs, targets)
        val_loss /= val_steps
        del val_loader
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
        print0(f"step:{step}/{train_steps} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step, 1):.2f}ms", console=True)
        
        model.train()
        # start the clock again
        torch.cuda.synchronize()
        t0 = time.perf_counter()

    if last_step:
        if master_process and args.save_checkpoint:
            log = dict(step=step, code=code, model=model.state_dict(), optimizer=optimizer.state_dict())
            os.makedirs(f"experiments/{run_id}", exist_ok=True) # Changed from "logs/" to "experiments/"
            torch.save(log, f"experiments/{run_id}/state_step{step:06d}.pt") # Changed from "logs/" to "experiments/"
        # the last step only has the validation loop, so break to avoid training
        break

    # --------------- TRAINING SECTION -----------------
    inputs, targets = next(train_loader)
    
    # Check if inputs exceed sequence length - can happen if the dataset has different sized examples
    if inputs.size(0) > args.train_seq_len:
        inputs = inputs[:args.train_seq_len]
        targets = targets[:args.train_seq_len]
        
    model(inputs, targets).backward()
    for param in model.parameters():
        dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
    # set optimization hyperparameters
    for group in optimizer.param_groups:
        group["lr"] = args.lr_init * get_lr(step)
    # step the optimizers
    optimizer.step()
    # Apply cosine normalization & absolute value constraints after optimization step
    model.enforce_constraints()  
    # null the gradients
    model.zero_grad(set_to_none=True)
        
    # logging
    approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
    print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms/(step + 1):.2f}ms", console=True)

print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
       f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)
dist.destroy_process_group()

# Then at the end of training:
if master_process:
    # check to make sure abs val & cos norm actually worked
    # checking to make sure the absolute value-ing worked
    print0("-"*10 + " making sure assertions worked " + "-"*10, console=True)
    print0(model.blocks[0].alpha_A.s.data[0,:5], console=True)
    # checking to make sure the cosine normalization worked
    print0(model.blocks[0].mlp.Wup.weight.norm(dim=1)[:5], console=True)
    print0(model.embed.weight.norm(dim=1)[:5], console=True)

    prompts = [
        "Once upon a time,",
        "The meaning of life is",
        "In the year 2026,",
        "I'm a Large Language Model (LLM), which means"
    ]
    for prompt in prompts:
        continuation = sample_from_model(model, prompt, max_new_tokens=16)
        print0(continuation, console=True)

########################################
#        HellaSwag Evaluation         #
########################################

def render_hellaswag_example(example, enc):
    """
    Given the example as a dictionary, render it as three torch tensors:
    - tokens (the tokens of context + completion, of size 4xN, as there are always 4 candidates)
    - mask (is 1 in the region of the candidate completion, where we evaluate likelihoods)
    - label (the index of the correct completion, which we hope has the highest likelihood)
    """
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    # gather up all the tokens
    ctx_tokens = enc.encode(ctx)
    tok_rows = []
    mask_rows = []
    for end in endings:
        end_tokens = enc.encode(" " + end)  # note: prepending " " because GPT-2 tokenizer
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0]*len(ctx_tokens) + [1]*len(end_tokens))

    # have to be careful during the collation because the number of tokens in each row can differ
    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.int32)
    mask = torch.zeros((4, max_len), dtype=torch.int32)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)

    return tokens, mask, label

def iterate_hellaswag_examples(data_path, limit=None):
    """Iterate through HellaSwag examples, with optional limit"""
    with open(data_path, "r") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            example = json.loads(line)
            yield example

@torch.no_grad()
def evaluate_hellaswag(model, data_path, limit=None):
    """Evaluate model on HellaSwag"""
    print0("Starting HellaSwag evaluation...", console=True)
    
    # Add this line at the beginning of the function to disable dynamo compilation for evaluation
    torch._dynamo.config.disable = True
    
    # Set up tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    model.eval()
    
    num_correct_norm = 0
    num_correct = 0
    num_total = 0
    
    for example in iterate_hellaswag_examples(data_path, limit):
        tokens, mask, label = render_hellaswag_example(example, enc)
        tokens = tokens.to(device="cuda")
        mask = mask.to(device="cuda")

        # Process each candidate one at a time to avoid memory issues
        losses = []
        normalized_losses = []
        
        for i in range(4):  # 4 candidates per example
            # Get token sequence for this candidate
            seq = tokens[i]
            seq_mask = mask[i]
            
            # Only process up to valid tokens (not padding)
            valid_len = (seq > 0).sum().item()
            if valid_len == 0:
                continue
                
            valid_seq = seq[:valid_len]
            
            # Get logits from our model
            logits = model(valid_seq)
            if isinstance(logits, torch.Tensor):
                logits = logits[0]  # Our model returns [B, T, V] but B=1
            
            # Evaluate the autoregressive loss
            shift_logits = logits[:-1, :]
            shift_tokens = valid_seq[1:].to(torch.int64)  # Target needs to be int64
            shift_mask = seq_mask[1:valid_len]  # Shift mask to align with shifted tokens
            
            # Calculate loss for each position
            losses_per_token = F.cross_entropy(
                shift_logits, shift_tokens, reduction='none'
            )
            
            # Apply mask to focus on completion region
            masked_losses = losses_per_token * shift_mask
            
            # Calculate total and normalized loss
            total_loss = masked_losses.sum()
            completion_token_count = shift_mask.sum()
            normalized_loss = total_loss / completion_token_count if completion_token_count > 0 else float('inf')
            
            losses.append(total_loss.item())
            normalized_losses.append(normalized_loss.item())
        
        # Get predictions
        pred = torch.tensor(losses).argmin().item()
        pred_norm = torch.tensor(normalized_losses).argmin().item()
        
        # Accumulate stats
        num_total += 1
        num_correct += int(pred == label)
        num_correct_norm += int(pred_norm == label)
        
        if num_total <= 5:  # Show details for first few examples
            print0(f"---\nContext:\n {example['ctx']}\nEndings:", console=True)
            for i, end in enumerate(example["endings"]):
                print0(f"{i} (loss: {normalized_losses[i]:.4f}) {end}", console=True)
            print0(f"predicted: {pred_norm}, actual: {label}", console=True)
    
    # Calculate accuracy
    accuracy = num_correct / num_total if num_total > 0 else 0
    accuracy_norm = num_correct_norm / num_total if num_total > 0 else 0
    
    # Calculate 95% confidence intervals using Wilson score interval
    # This is more robust than normal approximation, especially for small sample sizes or extreme probabilities
    z = 1.96  # 95% confidence
    
    def wilson_conf_interval(correct, total):
        """Calculate Wilson score interval for a binary proportion"""
        if total == 0:
            return (0, 0)
        
        p = correct / total
        denominator = 1 + z**2 / total
        centre_adjusted_p = (p + z**2 / (2 * total)) / denominator
        adjusted_interval = z * ((p * (1 - p) / total + z**2 / (4 * total**2)) ** 0.5) / denominator
        
        lower = max(0, centre_adjusted_p - adjusted_interval)
        upper = min(1, centre_adjusted_p + adjusted_interval)
        
        return (lower, upper)
    
    # Get confidence intervals
    ci = wilson_conf_interval(num_correct, num_total)
    ci_norm = wilson_conf_interval(num_correct_norm, num_total)
    
    # Final results
    print0(f"HellaSwag evaluation complete - {num_total} examples", console=True)
    print0(f"Accuracy: {num_correct}/{num_total}={accuracy:.4f} [95% CI: {ci[0]:.4f}-{ci[1]:.4f}]", console=True)
    print0(f"Normalized accuracy: {num_correct_norm}/{num_total}={accuracy_norm:.4f} [95% CI: {ci_norm[0]:.4f}-{ci_norm[1]:.4f}]", console=True)

# After training and sample generations, evaluate on HellaSwag
if master_process:
    hellaswag_path = "./data/hellaswag/hellaswag_val.jsonl"  # Adjust path as needed
    
    # Check if the HellaSwag data file exists
    if os.path.exists(hellaswag_path):
        print0(f"Found HellaSwag dataset at {hellaswag_path}, running evaluation...", console=True)
        evaluate_hellaswag(model, hellaswag_path, limit=20)
    else:
        print0(f"HellaSwag dataset not found at {hellaswag_path}, skipping evaluation.", console=True)
