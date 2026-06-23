import json
from tokenizers import Tokenizer

tok = Tokenizer.from_file("my_tokenizer.json")

# Look at the raw merge rules that were learned, in order
data = json.loads(open("my_tokenizer.json").read())
merges = data["model"]["merges"]
print("=== First 20 merge rules learned (in order) ===")
for i, m in enumerate(merges[:20]):
    print(f"{i+1:2d}. {m}")

print("\n=== Full learned vocabulary (token -> id) ===")
vocab = tok.get_vocab()
for tok_str, idx in sorted(vocab.items(), key=lambda x: x[1]):
    print(f"{idx:3d}  {tok_str!r}")
