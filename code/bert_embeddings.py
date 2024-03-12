import os
import argparse
from copy import deepcopy
from collections import defaultdict
import smart_open

import numpy as np
import gensim
from transformers import BertModel, BertForMaskedLM, BertTokenizer, get_linear_schedule_with_warmup, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm


def read_corpus(file_path):
    """
    Generator that yields sentences from a corpus file, splitting by lines and then by spaces.
    
    :param file_path: Path to the corpus file.
    :return: Yields lists of words in each line.
    """
    with open(file_path) as f:
        for line in f:
            yield line.strip().split()


class TextDataset(Dataset):
    """
    PyTorch Dataset class for text data, tokenized and encoded for BERT.

    :param texts: List of sentences to include in the dataset.
    :param tokenizer: Tokenizer to use for encoding the text.
    :param max_length: Maximum token length for BERT encoding.
    """
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }


def chunk_text(text, tokenizer, max_length=512):
    """
    Splits a text into chunks that fit within the specified maximum length for BERT inputs.
    
    :param text: Text (list of words) to chunk.
    :param tokenizer: Tokenizer to use for determining word token lengths.
    :param max_length: Maximum length of tokens in a chunk.
    :return: List of text chunks, each represented as a single string.
    """
    words = text
    word_token_len = [len(tokenizer.tokenize(word)) for word in words]

    chunks = []
    chunk = []
    curr_len = 0
    for word, len_ in zip(words, word_token_len):
        if curr_len + len_ + 2 > max_length:
            chunks.append(" ".join(chunk))
            chunk = []
            curr_len = 0
        chunk.append(word)
        curr_len += len_

    if chunk:
        chunks.append(" ".join(chunk))

    return chunks


def train(model, data_collator, dataset, batch_size, lr, epochs, output_dir):
    """
    Trains the BERT model using the provided DataLoader, learning rate, number of epochs, and device.

    :param model: The BERT model to train.
    :param dataloader: DataLoader providing batches of training data.
    :param lr: Learning rate for the optimizer.
    :param epochs: Number of training epochs.
    """
    # optimizer = AdamW(model.parameters(), lr)
    # steps = len(dataloader) * epochs
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=steps)

    # for epoch in range(epochs):
    #     model.train()
    #     total_loss = 0
        
    #     progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False, unit="batch")
        
    #     for batch in progress_bar:
    #         inputs, masks = batch["input_ids"], batch["attention_mask"]
    #         inputs = inputs.to(device)
    #         masks = masks.to(device)

    #         model.zero_grad()

    #         outputs = model(input_ids=inputs, attention_mask=masks, labels=inputs)
    #         loss = outputs.loss
    #         total_loss += loss.item()

    #         loss.backward()
    #         optimizer.step()
    #         scheduler.step()
            
    #         progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
    #     avg_train_loss = total_loss / len(dataloader)
    #     print(f"Epoch {epoch + 1}/{epochs} | Average Training Loss: {avg_train_loss:.4f}")
    model_dir = os.path.join(output_dir, model.__class__.__name__) 
    log_dir = os.path.join(model_dir, "logs") 

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True) 
    
    training_args = TrainingArguments(
        output_dir=model_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        logging_dir=log_dir,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    trainer.train()


def adapt(params, model_name, ang_file_path, eng_file_path, output_dir):
    """
    Adapts BERT for medieval Latin corpora by further pretraining on provided text files, with customizable parameters.

    :param params: Dictionary containing training parameters such as batch size, learning rate, epochs, and device.
    :param model_name: Name of the BERT model to use.
    :param ang_file_path: Path to the file containing the Anglo-Saxon text corpus.
    :param eng_file_path: Path to the file containing the English text corpus.
    :param output_dir: Directory to save the adapted model and tokenizer.
    """
    batch_size = params.get('batch_size', 8)
    lr = params.get('lr', 5e-5)
    epochs = params.get('epochs', 1)
    device = params.get('device', torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    # load model
    try:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForMaskedLM.from_pretrained(model_name).to(device)
    except Exception as e:
        raise RuntimeError(f"Failed to load model or tokenizer: {e}")

    # process corpora
    ang_corpus = list(read_corpus(ang_file_path))
    eng_corpus = list(read_corpus(eng_file_path))

    ang_chunks = [chunk for doc in ang_corpus for chunk in chunk_text(doc, tokenizer)]
    eng_chunks = [chunk for doc in eng_corpus for chunk in chunk_text(doc, tokenizer)]
    chunks = deepcopy(ang_chunks)
    chunks.extend(eng_chunks)

    # prepare dataset and dataloader
    dataset = TextDataset(chunks, tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    # train and save model
    train(model, data_collator=data_collator, dataset=dataset, batch_size=batch_size, lr=lr, epochs=epochs, output_dir=output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


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
    for chunk in chunk:
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
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model.eval()

    # generate embeddings for each corpus
    ang_embeddings = generate_embeddings(ang_file_path, model, tokenizer)
    eng_embeddings = generate_embeddings(eng_file_path, model, tokenizer)

    # save embeddings
    ang_path = os.join(output_dir, "ang_embeddings.bin")
    eng_path = os.join(output_dir, "eng_embeddings.bin")
    save_embeddings(ang_embeddings, ang_path)
    save_embeddings(eng_embeddings, eng_path)


def main(args):
    params = {
        "batch_size": args.batch,
        "lr": args.lr,
        "epochs": args.epochs,
        "device": torch.device(args.device)
    }

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    if not os.path.isfile(args.ang):
        raise FileNotFoundError(f"Anglo-Saxon corpus file not found at {args.ang}")
    if not os.path.isfile(args.eng):
        raise FileNotFoundError(f"English corpus file not found at {args.eng}")

    # further pretrain BERT model
    adapt(params, args.model, args.ang, args.eng, args.out)

    # extract embeddings
    extract_embeddings(args.out, args.ang, args.eng, args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adapt BERT for medieval Latin corpora and extract embeddings.")
    parser.add_argument("--ang", type=str, default="inputs/AngOrdtext", help="Path to the Anglo-Saxon corpus file.")
    parser.add_argument("--eng", type=str, default="inputs/EngOrdtext", help="Path to the English corpus file.")
    parser.add_argument("--out", type=str, default="outputs/", help="Directory to save the adapted model, tokenizer, and embeddings.")
    parser.add_argument("--model", type=str, default="bert-base-multilingual-cased", help="BERT model to use.")
    parser.add_argument("--batch", type=int, default=8, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Training device (cuda or cpu).")

    args = parser.parse_args()
   
    main(args)

