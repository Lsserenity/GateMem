import time
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union, Any

import torch
from torch.utils.data import DataLoader, RandomSampler


@dataclass
class TrainConfig:
    device: str = "auto"          # "auto" / "cuda" / "cpu"
    batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = True

    learning_rate: float = 3e-4
    betas: Tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    max_iters: int = 1000
    log_every: int = 100


def _pick_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def train_one(
    model,
    train_dataset,
    cfg: TrainConfig,
    *,
    dc_memory=None,
    # dc 接口假设：dc(prompts: list[str]) -> Any 或 dc.update(prompts)
    # 你可以在下面对应位置改成你的函数名
):
    """
    自定义训练 loop：
    - DataLoader 产出 x,y (以及可选 prompts)
    - 喂给 model(x, y, dc_memory=...) 或 model(x, y)
    """

    device = _pick_device(cfg.device)
    model = model.to(device)
    model.train()
    print("running on device", device)

    # ---- optimizer：优先用 model.configure_optimizers（minGPT 的习惯），否则退化到 AdamW ----
    optimizer = None
    if hasattr(model, "configure_optimizers"):
        # 伪造一个带必要字段的 config（兼容 minGPT 的 configure_optimizers）
        opt_cfg = type("OptCfg", (), {
            "learning_rate": cfg.learning_rate,
            "betas": cfg.betas,
            "weight_decay": cfg.weight_decay,
        })()
        optimizer = model.configure_optimizers(opt_cfg)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            betas=cfg.betas,
            weight_decay=cfg.weight_decay,
        )

    # ---- DataLoader：replacement=True + huge num_samples = 近似“无限流” ----
    loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        shuffle=False,
        sampler=RandomSampler(train_dataset, replacement=True, num_samples=int(1e10)),
    )
    data_iter = iter(loader)

    t_last = time.time()
    for it in range(1, cfg.max_iters + 1):
        # ---- fetch batch ----
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        # batch 可能是 (x, y) 或 (x, y, prompts)
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            x = batch[0]
            y = batch[1]
            prompts = batch[2] if len(batch) >= 3 else None
        else:
            raise RuntimeError(f"Unexpected batch type/structure: {type(batch)}")

        # x,y 是 tensor，prompts 是 list[str]（如果你 dataset 返回了）
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # ---- DC：输入 (batchsize, prompt) ----
        # 你说 DC 不可微，那就别让它进 autograd
        # 这一步只做 side effect（更新 DC）或产出一个非梯度特征
        dc_memory = None
        if dc is not None and prompts is not None:
            with torch.no_grad():
                # 下面三种你选一种用（按你 DC 的接口改名）：
                # 1) dc_memory = dc(prompts)
                # 2) dc_memory = dc.read(prompts)
                # 3) dc.update(prompts); dc_memory = dc  (把 dc 自己当 memory 传入)
                if hasattr(dc, "read"):
                    dc_memory = dc.read(prompts)
                elif hasattr(dc, "update"):
                    dc.update(prompts)
                    dc_memory = dc
                else:
                    dc_memory = dc(prompts)

        # ---- forward & loss ----
        # 兼容你 model.forward(x, y, dc_memory=...)
        out = None
        try:
            out = model(x, y, dc_memory=dc_memory)
        except TypeError:
            # 如果你的 forward 不支持 dc_memory，就退化成 model(x, y)
            out = model(x, y)

        # minGPT 通常返回 (logits, loss)
        if isinstance(out, (tuple, list)) and len(out) >= 2:
            logits, loss = out[0], out[1]
        else:
            raise RuntimeError("Model forward should return (logits, loss) or tuple-like output.")

        # ---- backward & step ----
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        # ---- logging ----
        if it % cfg.log_every == 0:
            t_now = time.time()
            dt = t_now - t_last
            t_last = t_now
            print(f"iter {it}/{cfg.max_iters} | loss {loss.item():.5f} | dt {dt:.2f}s")

    print("Training complete.")
