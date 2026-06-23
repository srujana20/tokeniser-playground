from tokenizers import Tokenizer

# Load tokenizer
tok = Tokenizer.from_file("my_tokenizer.json")

print("=" * 60)
print(f"Tokenizer Vocabulary Size: {tok.get_vocab_size()}")
print("=" * 60)

sentences = [
    "the cat sat on the mat",
    "the elephant tokenization expert",
    "byte pair encoding",
    "xyzzy plugh",
]

for s in sentences:
    enc = tok.encode(s)

    print("\n" + "=" * 60)
    print(f"Input: {s!r}")
    print("-" * 60)

    print(f"Character Count : {len(s)}")
    print(f"Token Count     : {len(enc.tokens)}")
    print(
        f"Avg Chars/Token : {len(s) / len(enc.tokens):.2f}"
        if enc.tokens else "N/A"
    )

    print("\nTokens:")
    print(enc.tokens)

    print("\nToken IDs:")
    print(enc.ids)

    print("\nToken Breakdown:")
    for token, token_id in zip(enc.tokens, enc.ids):
        print(
            f"  {token!r:<20} "
            f"ID={token_id:<4} "
            f"Length={len(token)}"
        )