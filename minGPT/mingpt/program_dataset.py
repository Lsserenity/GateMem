import json
import torch
from torch.utils.data import Dataset

class ProgramDataset(Dataset):
    """
    jsonl 每行: {"x": prompt, "y": completion}
    返回:
      idx:     [block_size]
      targets: [block_size]，并 mask 掉 x 部分，只在 y 上算 loss
    """

    def __init__(self, jsonl_path, tokenizer, block_size, pad_token_id=0):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.pad_token_id = pad_token_id

        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                self.samples.append((ex["x"], ex["y"]))

    def __len__(self):
        return len(self.samples)

    def _pad_to_block(self, t: torch.Tensor, pad_value: int):
        # t: [T]
        if t.numel() >= self.block_size:
            return t[: self.block_size]
        pad_len = self.block_size - t.numel()
        pad = torch.full((pad_len,), pad_value, dtype=t.dtype)
        return torch.cat([t, pad], dim=0)

    def __getitem__(self, i):
        x, y = self.samples[i]
        prefix = x + " "
        full = prefix + y

        idx_full = self.tokenizer(full)[0]      # [T]
        idx_prefix = self.tokenizer(prefix)[0]  # [Tx]

        idx_full = idx_full[: self.block_size]

        targets = idx_full.clone()
        targets[:-1] = idx_full[1:]
        targets[-1] = -1

        start_y = min(len(idx_prefix), len(idx_full))
        loss_start = max(start_y - 1, 0)
        targets[:loss_start] = -1

        # pad 到固定长度
        idx_full = self._pad_to_block(idx_full, pad_value=self.pad_token_id)
        targets  = self._pad_to_block(targets,  pad_value=-1)

        return idx_full, targets, x, y
