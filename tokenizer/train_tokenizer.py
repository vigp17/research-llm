import os
import sentencepiece as spm
from datasets import load_dataset
from tqdm import tqdm


def train_tokenizer():
    output_dir = "tokenizer"
    os.makedirs(output_dir, exist_ok=True)

    corpus_path = os.path.join(output_dir, "tokenizer_corpus.txt")
    model_prefix = os.path.join(output_dir, "researchllm_bpe")

    vocab_size = 32000
    max_lines = 5_000_000  # Enough for a high-quality tokenizer

    print("🔹 Streaming dataset to build tokenizer corpus...")

    dataset = load_dataset(
        "allenai/c4",
        "en",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    with open(corpus_path, "w", encoding="utf-8") as f:
        for i, example in enumerate(tqdm(dataset)):
            if i >= max_lines:
                break
            text = example.get("text", "")
            text = text.replace("\x00", "")
            if text.strip():
                f.write(text.replace("\n", " ") + "\n")

    print("✅ Corpus built. Training SentencePiece BPE tokenizer...")

    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=0.9995,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
    )

    os.remove(corpus_path)
    print("🎉 Tokenizer training complete!")
    print(f"Saved: {model_prefix}.model")


if __name__ == "__main__":
    train_tokenizer()