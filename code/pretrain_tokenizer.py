import os
import json
from tokenizers import BertWordPieceTokenizer
from utils import *

def train_tokenizer(input_file, output_dir):
    """
    Trains a BertWordPieceTokenizer on a given corpus and saves the tokenizer model along with its configuration.

    Parameters:
    - input_file: Path to the text file or files containing the training corpus. Can be a single string (path to a file) or a list of strings (paths to multiple files).
    - output_dir: Directory where the trained tokenizer model and its configuration will be saved. If the directory does not exist, it will be created.
    """
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<S>", "<T>"]
    vocab_size = 32000
    max_length = 512
    truncate_longer_samples = False

    # train tokenizer
    tokenizer = BertWordPieceTokenizer(lowercase=True) 
    tokenizer.train(files=input_file, vocab_size=vocab_size, special_tokens=special_tokens, min_frequency=5)

    if truncate_longer_samples:
        tokenizer.enable_truncation(max_length=max_length)

    model_path = os.path.join(output_dir, "pretrained-tokenizer")
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    tokenizer.save_model(model_path)

    with open(os.path.join(model_path, "tokenizer_config.json"), "w") as f:
        tokenizer_cfg = {
            "do_lower_case": True,
            "unk_token": "[UNK]",
            "sep_token": "[SEP]",
            "pad_token": "[PAD]",
            "cls_token": "[CLS]",
            "mask_token": "[MASK]",
            "model_max_length": max_length,
            "max_len": max_length,
        }
        json.dump(tokenizer_cfg, f)