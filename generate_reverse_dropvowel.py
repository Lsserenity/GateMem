# generate_reverse_dropvowel.py
import json
import random
import string
from pathlib import Path

VOWELS = set("aeiouAEIOU")

def transform(s: str) -> str:
    """Reverse the string and remove vowels."""
    rev = s[::-1]
    return "".join(ch for ch in rev if ch not in VOWELS)

def rand_string(rng: random.Random, min_len=6, max_len=16) -> str:
    L = rng.randint(min_len, max_len)
    alphabet = string.ascii_lowercase  # 只用小写更稳
    return "".join(rng.choice(alphabet) for _ in range(L))

def make_example(inp: str) -> dict:
    x = (
        "Task: Reverse the string and remove vowels (a,e,i,o,u).\n"
        f"Input: {inp}\n"
        "Output:"
    )
    y = transform(inp)
    return {"x": x, "y": y}

def generate_dataset(
    out_dir: str = "data_reverse_dropvowel",
    n_train: int = 50000,
    n_test: int = 2000,
    seed: int = 42,
):
    rng = random.Random(seed)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 为了避免 train/test 重复输入字符串，显式去重
    seen = set()

    def gen_unique(n):
        items = []
        while len(items) < n:
            s = rand_string(rng)
            if s in seen:
                continue
            seen.add(s)
            items.append(s)
        return items

    train_inputs = gen_unique(n_train)
    test_inputs  = gen_unique(n_test)

    train_file = out_path / "train.jsonl"
    test_file  = out_path / "test.jsonl"

    with train_file.open("w", encoding="utf-8") as f:
        for s in train_inputs:
            f.write(json.dumps(make_example(s), ensure_ascii=False) + "\n")

    with test_file.open("w", encoding="utf-8") as f:
        for s in test_inputs:
            f.write(json.dumps(make_example(s), ensure_ascii=False) + "\n")

    print(f"Wrote: {train_file} ({n_train} examples)")
    print(f"Wrote: {test_file} ({n_test} examples)")
    print("Sample:", make_example("transformer"))

if __name__ == "__main__":
    generate_dataset()
