import os
from collections import defaultdict
import smart_open

import numpy as np
import gensim
import torch
from transformers import BertTokenizerFast, BertModel

from utils import *

def generate_embeddings(file_path, model, tokenizer):
    """
    Generate embeddings for a given corpus using the specified BERT model and tokenizer.
    
    :param file_path: Path to file containing the input corpus
    :param model: Loaded BERT model.
    :param tokenizer: Loaded BERT tokenizer.
    :return: Dictionary of averaged embeddings for each token in the corpus.
    """
    all_embeddings = defaultdict(list)

    # extract layers from each model output
    corpus = list(read_corpus(file_path))
    chunks = [chunk for doc in corpus for chunk in chunk_text(doc, tokenizer)]
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors='pt', padding='max_length', truncation=True, max_length=512, return_attention_mask=True)
        input_ids = inputs['input_ids']
        att_mask = inputs['attention_mask']

        with torch.no_grad():
            outputs = model(input_ids, attention_mask = att_mask, output_hidden_states=True)

        layers = sum(outputs.hidden_states[-4:])

        for token_id, token_embedding in zip(input_ids[0], layers[0]):
            token = tokenizer.decode([token_id])
            all_embeddings[token].append(token_embedding.numpy())

    # average embeddings for each token
    averaged_embeddings = {token: np.mean(embeddings, axis=0) for token, embeddings in all_embeddings.items()}
    return averaged_embeddings


def save_embeddings(embeddings, file_path):
    """
    Save embeddings in a binary format compatible with Gensim.
    
    :param embeddings: Dictionary of token embeddings.
    :param file_path: Path to save the binary file.
    """
    with smart_open.open(file_path, 'wb') as fout:
        for token, embedding in embeddings.items():
            fout.write(gensim.utils.to_utf8(f"{token} ") + embedding.tobytes() + b"\n")


def extract_embeddings(model_path, ang_file_path, eng_file_path, output_dir):
    """
    Extracts time-specific static embeddings for Anglo-Saxon and Norman period corpora.
    
    :param model_path: Path to the fine-tuned BERT model.
    :param ang_file_path: Path to the file containing the Anglo-Saxon text corpus.
    :param eng_file_path: Path to the file containing the English text corpus.
    """
    # load model
    model = BertModel.from_pretrained(model_path)
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    model.eval()

    # generate embeddings for each corpus
    ang_embeddings = generate_embeddings(ang_file_path, model, tokenizer)
    eng_embeddings = generate_embeddings(eng_file_path, model, tokenizer)

    # save embeddings
    ang_path = os.join(output_dir, "ang_embeddings.bin")
    eng_path = os.join(output_dir, "eng_embeddings.bin")
    save_embeddings(ang_embeddings, ang_path)
    save_embeddings(eng_embeddings, eng_path)