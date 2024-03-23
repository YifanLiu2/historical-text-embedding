from transformers import BertModel, BertTokenizer, BertTokenizerFast
import torch
import numpy as np
from tqdm import tqdm
import os
import argparse

corpus_name_reference = {"AngText": "ang", "EngText": "eng"}


def read_file(file_path):
    """Reads text data from a file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.readlines()


def get_absolute_file_paths(folder_path):
    """Extracts and returns absolute file paths."""
    return [
        os.path.join(folder_path, file)
        for file in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, file))
    ]


def save_embeddings_to_vec_file(word_embeddings, file_path):
    """
    Save the word embeddings dictionary to a .vec file.

    Args:
    - word_embeddings: A dictionary with words as keys and embeddings as values.
    - file_path: Path to the .vec file to save.
    """
    # Count the number of embeddings and the size of each embedding
    num_embeddings = len(word_embeddings)
    embedding_size = len(next(iter(word_embeddings.values())))

    with open(file_path, "w") as file:
        # Write the header with the number of embeddings and their size
        file.write(f"{num_embeddings} {embedding_size}\n")

        # Write each word and its embedding
        for word, embedding in word_embeddings.items():
            embedding_str = " ".join(map(str, embedding))
            file.write(f"{word} {embedding_str}\n")


# Function to aggregate token embeddings into word embeddings
def aggregate_token_embeddings_to_words(tokenizer, model, corpus, device):
    word_embeddings_sum = {}
    word_counts = {}

    with torch.no_grad():
        for sentence in tqdm(corpus, desc="Processing sentences"):
            # Encode the inputs and move them to the same device as the model
            inputs = tokenizer(
                sentence,
                return_tensors="pt",
                padding=True,
                truncation=True,
                return_offsets_mapping=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            offset_mapping = (
                inputs.pop("offset_mapping").cpu().numpy()[0]
            )  # Adjusted for fast tokenizer

            outputs = model(**inputs)
            hidden_states = outputs.hidden_states

            # Average the last four layers
            embeddings = torch.mean(torch.stack(hidden_states[-4:]), dim=0).squeeze(0)
            embeddings = embeddings.to("cpu").numpy()

            # Process each token
            for i, (start, end) in enumerate(offset_mapping):
                if start == end:  # Skip special tokens
                    continue
                word = sentence[start:end].lower()
                if word in word_embeddings_sum:
                    word_embeddings_sum[word] += embeddings[i]
                    word_counts[word] += 1
                else:
                    word_embeddings_sum[word] = embeddings[i]
                    word_counts[word] = 1
    # Calculate the average embedding for each word
    word_embeddings_avg = {
        word: word_embeddings_sum[word] / count for word, count in word_counts.items()
    }
    return word_embeddings_avg


def get_word_embeddings(tokenizer_path, bert_path, corpus_path, save_path):
    # If there's a GPU available...
    if torch.cuda.is_available():

        # Tell PyTorch to use the GPU.
        device = torch.device("cuda")

        print("There are %d GPU(s) available." % torch.cuda.device_count())
        print("We will use the GPU:", torch.cuda.get_device_name(0))

    # If not...
    else:
        print("No GPU available, using the CPU instead.")
        device = torch.device("cpu")
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
    corpus_path_lst = get_absolute_file_paths(corpus_path)
    model_path_lst = get_absolute_file_paths(bert_path)
    for corpus_path in corpus_path_lst:
        corpus = read_file(corpus_path)
        corpus_name = os.path.basename(corpus_path).split(".")[0]
        prefix = corpus_name_reference[corpus_name]
        for model_path in model_path_lst:
            model_name = os.path.basename(model_path).split(".")[0]
            model = BertModel.from_pretrained(
                model_path, output_hidden_states=True
            )  # Ensure the model outputs hidden states
            # Check if CUDA (GPU support) is available and move the model to GPU if it is
            model = model.to(device)
            model.eval()
            word_embeddings_avg = aggregate_token_embeddings_to_words(
                tokenizer=tokenizer, model=model, corpus=corpus, device=device
            )
            model_save_path = os.path.join(
                save_path, f"{prefix}_{model_name}_embedding.vec"
            )
            save_embeddings_to_vec_file(word_embeddings_avg, model_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract word embeddings using BERT.")
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        required=True,
        help="Path to the pretrained tokenizer.",
    )
    parser.add_argument(
        "--bert_path",
        type=str,
        required=True,
        help="Directory containing pretrained BERT models.",
    )
    parser.add_argument(
        "--corpus_path",
        type=str,
        required=True,
        help="Directory containing corpus files.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Directory to save the extracted embeddings.",
    )

    args = parser.parse_args()
    get_word_embeddings(
        args.tokenizer_path, args.bert_path, args.corpus_path, args.save_path
    )
