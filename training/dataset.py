import torch
from torch.utils.data import IterableDataset
import sentencepiece as spm
from datasets import load_dataset


class StreamingTokenDataset(IterableDataset):
    def __init__(
        self,
        tokenizer_path: str,
        seq_len: int = 1024,
        split: str = "train",
    ):
        super().__init__()

        self.seq_len = seq_len

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(tokenizer_path)

        # Stream dataset (no RAM explosion)
        self.dataset = load_dataset(
            "togethercomputer/RedPajama-Data-1T-Sample",
            split=split,
            streaming=True,
        )

    def __iter__(self):
        buffer = []

        for example in self.dataset:
            text = example.get("text", "")
            if not text:
                continue

            tokens = self.sp.encode(text)

            buffer.extend(tokens)

            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[: self.seq_len + 1]
                buffer = buffer[self.seq_len :]

                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)

                yield x, y