# tokeniser-playground

Small playground for training and inspecting a BPE tokenizer.

Repository files

- `corpus.txt` — training corpus (one or more lines of text).
- `train_bpe.py` — script to train a BPE tokenizer from `corpus.txt`.
- `encode_demo.py` — demo script to encode text with the trained tokenizer.
- `inspect_tokenizer.py` — helper to inspect `my_tokenizer.json` or other tokenizer artifacts.
- `my_tokenizer.json` — example/output tokenizer file (JSON format).

Quick start

1. Ensure you have Python 3.8+ installed.
2. (Optional) Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate
```

3. Install common dependencies (if your scripts require any — adjust as needed):

```bash
pip install -r requirements.txt || pip install tokenizers tqdm
```

Train tokenizer

```bash
python train_bpe.py
```

This should read `corpus.txt` and write a tokenizer file (for example `my_tokenizer.json`). Check `train_bpe.py` for options and parameters.

Encode text with the tokenizer

```bash
python encode_demo.py "This is a test sentence."
```

Inspect the tokenizer

```bash
python inspect_tokenizer.py my_tokenizer.json
```

Notes

- The exact command-line flags and behavior depend on the scripts in this repo. Open `train_bpe.py`, `encode_demo.py`, and `inspect_tokenizer.py` to see available options.
- If you want, I can add a `requirements.txt`, example corpus, or expand the README with usage examples and expected outputs.
