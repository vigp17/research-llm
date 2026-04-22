import sentencepiece as spm

def test_tokenizer():
    sp = spm.SentencePieceProcessor()
    sp.load("tokenizer/researchllm_bpe.model")

    text = "ResearchLLM is a transformer language model trained from scratch."
    tokens = sp.encode(text, out_type=int)
    decoded = sp.decode(tokens)

    print("Original text:")
    print(text)
    print("\nToken IDs:")
    print(tokens)
    print("\nDecoded text:")
    print(decoded)


if __name__ == "__main__":
    test_tokenizer()