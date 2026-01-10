from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.bpe import BPETokenizer
from mingpt.program_dataset import ProgramDataset

def main():
    tokenizer = BPETokenizer()

    cfg = GPT.get_default_config()
    cfg.model_type = "gpt-nano"
    cfg.vocab_size = 50257
    cfg.block_size = 128

    model = GPT(cfg, types="nm")  # 你的 memory 版本

    train_ds = ProgramDataset(
        jsonl_path="../data_reverse_dropvowel/train.jsonl",
        tokenizer=tokenizer,
        block_size=cfg.block_size,
    )

    tcfg = Trainer.get_default_config()
    tcfg.batch_size = 32
    tcfg.learning_rate = 3e-4
    tcfg.max_iters = 2000
    tcfg.num_workers = 0

    trainer = Trainer(tcfg, model, train_ds)
    trainer.run()

if __name__ == "__main__":
    main()


"""
新建的训练入口
先用 gpt-nano 跑通；之后你再换 gpt2/冻结参数
"""