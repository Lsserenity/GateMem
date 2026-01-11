"""
Full definition of a GPT Language Model, all of it in this single file.

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from mingpt.utils import CfgNode as CN

# -----------------------------------------------------------------------------

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

# ================ 开始修改 ===================
class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config, nm = None):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            act     = NewGELU(),
            dropout = nn.Dropout(config.resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # MLP forward

        # # 引入neural memory模块
        # self.nm = nm
        # if nm is not None:
        #     self.dc_gate = nn.Sequential(
        #         nn.Linear(config.n_embd, 4 * config.n_embd),
        #         nn.SiLU(),
        #         nn.Linear(4 * config.n_embd, config.n_embd),
        #         nn.Sigmoid()
        #     )
        
        #     self.gate = nn.Sequential(
        #         nn.Linear(config.n_embd, 4 * config.n_embd),
        #         nn.SiLU(),
        #         nn.Linear(4 * config.n_embd, config.n_embd),
        #         nn.Sigmoid()
        #     )


    def forward(self, x): # forward(self, x, dc_memory = None)
        # 查询neural memory
        # if self.nm is not None:
        #     B, T, C = x.shape
        #     nm_memory = self.nm.retrieve(x.reshape(B*T, C)).reshape(B, T, C)
        #     if dc_memory is None:
        #         dc_memory = torch.zeros_like(x)
        #     g = self.dc_gate(dc_memory)
        #     x = x + g * nm_memory

        #     x = x + self.attn(self.ln_1(x))
        #     x = x + self.mlpf(self.ln_2(x))

        #     # 更新neural memory
        #     self.nm.update(x.reshape(B*T, C))
        
        #     # 再次查询neural memory进行融合
        #     nm_memory = self.nm.retrieve(x.reshape(B*T, C), new_params = self.nm.new_params).reshape(B, T, C)
        #     x = x + self.gate(x) * nm_memory
        # else:
        #     x = x + self.attn(self.ln_1(x))
        #     x = x + self.mlpf(self.ln_2(x))

        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))

        return x

class GPT(nn.Module):
    """ GPT Language Model """

    @staticmethod
    def get_default_config():
        C = CN()
        # either model_type or (n_layer, n_head, n_embd) must be given in the config
        C.model_type = 'gpt'
        C.n_layer = None
        C.n_head = None
        C.n_embd =  None
        # these options must be filled in externally
        C.vocab_size = None
        C.block_size = None
        # dropout hyperparameters
        C.embd_pdrop = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1

        C.nm_use_post_read = True  # 是否在transformer最后使用post-read
        return C

    def __init__(self, config, types = None):
        super().__init__()
        self.types = types
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.block_size = config.block_size

        type_given = config.model_type is not None
        params_given = all([config.n_layer is not None, config.n_head is not None, config.n_embd is not None])
        assert type_given ^ params_given # exactly one of these (XOR)
        if type_given:
            # translate from model_type to detailed configuration
            config.merge_from_dict({
                # names follow the huggingface naming conventions
                # GPT-1
                'openai-gpt':   dict(n_layer=12, n_head=12, n_embd=768),  # 117M params
                # GPT-2 configs
                'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
                'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
                'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
                'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
                # Gophers
                'gopher-44m':   dict(n_layer=8, n_head=16, n_embd=512),
                # (there are a number more...)
                # I made these tiny models up
                'gpt-mini':     dict(n_layer=6, n_head=6, n_embd=192),
                'gpt-micro':    dict(n_layer=4, n_head=4, n_embd=128),
                'gpt-nano':     dict(n_layer=3, n_head=3, n_embd=48),
            }[config.model_type])

        # neural network 添加
        if self.types is not None:
            from .memory_network import NeuralMemory
            self.neural_memory = NeuralMemory(
                input_dim = config.n_embd,
                layers = 2,
                hidden_dim = 4 * config.n_embd,
            )
        else:
            self.neural_memory = None

        if self.neural_memory is not None:
            self.dc_gate = nn.Sequential(
                nn.Linear(config.n_embd, 4 * config.n_embd),
                nn.SiLU(),
                nn.Linear(4 * config.n_embd, config.n_embd),
                nn.Sigmoid()
            )
            self.nm_gate = nn.Sequential(
                nn.Linear(config.n_embd, 4 * config.n_embd),
                nn.SiLU(),
                nn.Linear(4 * config.n_embd, config.n_embd),
                nn.Sigmoid()
            )
        else:
            self.dc_gate = None
            self.nm_gate = None


        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.embd_pdrop),
            # # 将Block中的neural memory模块传入
            # h = nn.ModuleList([Block(config, nm = self.neural_memory) for _ in range(config.n_layer)]),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        assert self.transformer.wte.embedding_dim == self.transformer.wpe.embedding_dim, \
        f"词嵌入维度({self.transformer.wte.embedding_dim})必须等于位置嵌入维度({self.transformer.wpe.embedding_dim})"
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    @classmethod
    def from_pretrained(cls, model_type, types = None, model_dir=None):
        """
        Initialize a pretrained GPT model by copying over the weights
        from a huggingface/transformers checkpoint.
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        import torch
        import os
        from transformers import GPT2LMHeadModel

        if model_dir is not None:
            assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
            hf_source = model_type
        else:
            hf_source = model_dir
            if not os.path.isdir(hf_source):
                raise FileNotFoundError(f"model_dir not found: {hf_source}")


        # create a from-scratch initialized minGPT model
        config = cls.get_default_config()
        config.model_type = model_type
        config.vocab_size = 50257 # openai's model vocabulary
        config.block_size = 1024  # openai's model block_size
        model = GPT(config, types = types)
        sd = model.state_dict()

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(
            hf_source,
            local_files_only=model_dir is not None
        )
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        keys = [k for k in sd_hf if not k.endswith('attn.masked_bias')] # ignore these
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla nn.Linear.
        # # this means that we have to transpose these weights when we import them
        # assert len(keys) == len(sd)
        # 修改，只复制transformer部分的参数
        for k in keys:
            if k not in sd:
                continue
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        missing = [k for k in sd.keys() if k not in sd_hf.keys()]

        return model, missing

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters(recurse=False):
                if not p.requires_grad:
                    continue # frozen weights
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    # 修改，传入dc_memory
    def forward(self, idx, targets=None, dc_memory = None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        # 防止batch之间的nm fast weights相互污染
        if self.neural_memory is not None and self.training:
            self.neural_memory.new_params = None

        # 最后3层注入：read + dc_gate
        n_layer = len(self.transformer.h)
        inject_layers = {n_layer - 3, n_layer - 2, n_layer - 1}

        for i, block in enumerate(self.transformer.h):

            if self.neural_memory is not None and i in inject_layers:
                B, T, C = x.shape

                # 1) pre-read：NM 决定读出什么
                m = self.neural_memory.retrieve(x.reshape(B * T, C)).reshape(B, T, C)

                # 2) DC 决定“信多少”（强度门）
                if dc_memory is None:
                    dc_vec = torch.zeros(B, 1, C, device=x.device)
                else:
                    # 兼容 dc_memory: (B, dc_len, C)
                    dc_vec = dc_memory.mean(dim=1, keepdim=True)

                g_dc = self.dc_gate(dc_vec).expand(B, T, C)

                # 3) （可选）再乘一个基于x的门（你已有 nm_gate，就先用上）
                g_nm = self.nm_gate(x)   # (B,T,C)
                x = x + (g_dc * g_nm) * m

            x = block(x)

        # write: transformer结束后写入NM（会更新 self.neural_memory.new_params）
        if self.neural_memory is not None:
            B, T, C = x.shape
            self.neural_memory.update(x.reshape(B * T, C))


        # head: ln_f + lm_head
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # loss & return
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )

        return logits, loss


    def generate(self, 
                 idx, 
                 max_new_tokens, 
                 temperature=1.0, 
                 do_sample=False, 
                 top_k=None, 
                 dc_memory = None,
                 eos_token_id = None,
                 return_only_generated=True):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        B, T0 = idx.size()
        finished = torch.zeros(B, dtype=torch.bool, device=idx.device)
        generated = []

        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # forward the model to get the logits for the index in the sequence
            # 修改，传入dc_memory！！！
            dc_mem = None if dc_memory is None else dc_memory
            logits, _ = self(idx_cond, dc_memory = dc_mem)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :].detach() / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            
            if eos_token_id is not None:
                # 如果生成了 eos_token_id，就停止生成
                idx_next = torch.where(
                    finished.unsqueeze(1),
                    torch.full_like(idx_next, eos_token_id),
                    idx_next
                )

            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
            generated.append(idx_next)

            if eos_token_id is not None:
                finished = finished | (idx_next.squeeze(1) == eos_token_id)
                if finished.all():
                    break
        
        generated = torch.cat(generated, dim=1) if len(generated) > 0 else idx.new_zeros((B, 0))

        if return_only_generated:
            return generated
        else:
            return idx