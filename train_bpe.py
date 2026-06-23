from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# A small training corpus -- in real life this would be billions of words
corpus = [
    "the cat sat on the mat",
    "the cat sat on the chair",
    "the dog ran in the park",
    "tokenization is the process of converting text into tokens",
    "large language models predict the next token",
    "the cat and the dog played together in the park",
    "byte pair encoding merges frequent character pairs",
    "this tokenizer is being trained from scratch right now",
]

# Write corpus to a file (the trainer reads from files)
with open("corpus.txt", "w") as f:
    f.write("\n".join(corpus))

# Set up a BPE tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()
print(tokenizer)
# vocab_size is tiny on purpose so we can actually see the merges happen
trainer = BpeTrainer(
    vocab_size=40,
    special_tokens=["[UNK]"],
    show_progress=False,
)

tokenizer.train(["corpus.txt"], trainer)

tokenizer.save("my_tokenizer.json")
print("Training complete. Vocab size:", tokenizer.get_vocab_size())
