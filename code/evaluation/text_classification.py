from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
import os
from tqdm import tqdm
from gensim.models import FastText
from transformers import BertTokenizer, BertModel
import argparse
from torch.utils.data import Dataset
from sklearn.model_selection import KFold
from torch.utils.data import Subset


def read_file(file_path):
    """Reads text data from a file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.readlines()


def read_labels(file_path):
    """Reads labels from a file."""
    with open(file_path, "r", encoding="utf-8") as file:
        labels = pd.Series([label for line in file for label in line.strip().split()])
        labels = labels.replace('NA', pd.NA)
        labels = labels.astype('category').cat.codes
    return labels.tolist()

def create_dataset(embeddings, labels, nan_value=-1):
    """
    Filters out data points with NaN labels and creates a TensorDataset with the remaining data.
    """
    valid_idx = [i for i, label in enumerate(labels) if label != nan_value]

    # Use valid indices to filter embeddings and labels
    embeddings = embeddings[valid_idx]
    labels = labels[valid_idx]

    return TensorDataset(embeddings, labels)
    

def k_fold_split(dataset, k=5):
    """
    randomly split a PyTorch tensor dataset into k folds.
    """
    indices = torch.randperm(len(dataset)).tolist()
    
    fold_sizes = len(dataset) // k
    folds = []
    
    for i in range(k):
        val_idx = indices[i*fold_sizes : (i+1)*fold_sizes]
        train_idx = indices[:i*fold_sizes] + indices[(i+1)*fold_sizes:]
        
        val_dataset = dataset[val_idx]
        train_dataset = dataset[train_idx]
        
        folds.append((train_dataset, val_dataset))
    
    return folds

class TextDataset(Dataset):
    def __init__(self, corpus, labels, model, tokenizer=None, is_bert=False, nan_value=-1):
        self.corpus = corpus
        self.labels = labels
        self.tokenizer = tokenizer
        self.model = model
        self.is_bert = is_bert

        # filter out NaN values
        self.corpus, self.labels = zip(*[(text, label) for text, label in zip(corpus, labels) if label != nan_value])
        self.labels = list(self.labels)
        self.corpus = list(self.corpus)


    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        text = self.corpus[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.is_bert:
            embedding = sentence_to_bert_embeddings(text, self.model, self.tokenizer)
        else:
            self.max_seq_length = max(len(text.split()) for text in self.corpus)
            embedding = sentence_to_embedding(text, self.model, self.max_seq_length)

        return embedding, label



###### Extract Features for Static Word Embeddings ######
def sentence_to_embedding(sentence, embeddings, max_seq_length):
    """Converts a sentence to an embedding."""

    embed_dim = embeddings.vector_size
    words = sentence.split()
    word_embeddings = np.zeros((max_seq_length, embed_dim))

    for i, word in enumerate(words):
        try:
            word_embeddings[i] = embeddings.wv[word]
        # handle OOV words
        except KeyError: 
            word_embeddings[i] = np.zeros(embed_dim)

    return torch.tensor(word_embeddings, dtype=torch.float32) #[seq_length, embed_size]


###### Extract Features for BERT ######
def sentence_to_bert_embeddings(sentence, model, tokenizer):
    # tokenize inputs
    inputs = tokenizer(sentence, padding=True, truncation=True, max_length=512, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs['attention_mask']

    # get embeddings from the second-to-last layer
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # Get the second-to-last layer embeddings
        hidden_states = outputs.hidden_states
        token_embeddings = hidden_states[-2]  # [1, seq_length, 768]

    return token_embeddings.squeeze(0) # [seq_length, 768]


class SequenceClassifier(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_classes):
        super(SequenceClassifier, self).__init__()
        
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True, dropout=0.3)
        self.dropout = nn.Dropout(0.3)
        lstm_output_dim = hidden_dim * 2
        self.fc = nn.Linear(lstm_output_dim, num_classes)

    def forward(self, embeddings):
        # embeddings: [batch_size, seq_length, embedding_dim]
        lstm_out, _ = self.lstm(embeddings)  # [batch_size, seq_length, lstm_output_dim]
        lstm_out = self.dropout(lstm_out)
        final = lstm_out[:, -1, :]  # [batch_size, lstm_output_dim]
        logits = self.fc(final)  # [batch_size, num_classes]
        
        return logits


def train(model, dataset, batch_size, epoch_num, learning_rate, device):
    """Train"""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epoch_num):
        model.train() 
        total_loss = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epoch_num}")

        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device) 
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            progress_bar.set_postfix({'loss': f'{total_loss / (progress_bar.last_print_n + 1):.4f}'})

        print(f"Epoch {epoch+1} completed, Average Loss: {total_loss / len(dataloader)}")


def evaluate(model, dataset, batch_size, device):
    """Evaluates model performance. """
    model.eval() 
    model.to(device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_predictions, all_true_labels = [], []

    with torch.no_grad():         
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=1)

            all_predictions.extend(predictions.detach().cpu().numpy())
            all_true_labels.extend(labels.detach().cpu().numpy())

    # calculate evaluation metrics
    num_classes = len(np.unique(all_true_labels))
    avg_type = "binary" if num_classes == 2 else "macro"
    accuracy = accuracy_score(all_true_labels, all_predictions)
    precision = precision_score(all_true_labels, all_predictions, average=avg_type, zero_division=0)
    recall = recall_score(all_true_labels, all_predictions, average=avg_type)
    f1 = f1_score(all_true_labels, all_predictions, average=avg_type, zero_division=0)

    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    return accuracy, precision, recall, f1


def main_eval_loop(model_path, is_bert, corpus_path, label_dir, output_dir, tokenizer_path=None, input_label=None):
    """Evaluates embeddings and model."""
    # ---- load data and label ----
    if input_label:
        label_paths = [(input_label, os.path.join(label_dir, input_label))]
    else:
        label_paths = [(file, os.path.join(label_dir, file)) for file in os.listdir(label_dir) if os.path.isfile(os.path.join(label_dir, file))]

    try:
        corpus = read_file(corpus_path)
    except FileNotFoundError:
        raise ValueError("Corpus file not found at the specified path.")

    # ---- load model ----
    print("load model ...")
    if is_bert:  # Load BERT model
        try:
            model = BertModel.from_pretrained(model_path)
            tokenizer = BertTokenizer.from_pretrained(tokenizer_path or model_path)
            embed_dim = model.config.hidden_size 
        except Exception as e:
            raise ValueError(f"Failed to load BERT model and tokenizer: {e}")
    else:  # Load FastText model
        try:
            model = FastText.load(model_path)
            embed_dim = model.wv.vector_size
        except Exception as e:
            raise ValueError(f"Failed to load FastText model: {e}")
    print(f"load the model from: {model_path}")

    # ---- train and evaluate ----
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_path = os.path.join(output_dir, "extrinsic_eval.txt")

    with open(output_path, "a") as f:
        for file_name, label_path in label_paths:
            try:
                labels = torch.tensor(read_labels(label_path), dtype=torch.long)
            except FileNotFoundError:
                print(f"Metadata label file not found: {label_path}")
                continue
            num_classes = len(torch.unique(labels))
            dataset = TextDataset(corpus, labels, model, tokenizer if is_bert else None, is_bert=is_bert)

            # 5 fold cross validation
            metrics = {'acc': [], 'prec': [], 'recall': [], 'f1': []}
            kf = KFold(n_splits=5)
            for i, (train_idx, val_idx) in enumerate(kf.split(dataset)):
                print(f"evaluate metadata: {file_name}, fold {i + 1}/5")
                train_dataset = Subset(dataset, train_idx)
                val_dataset = Subset(dataset, val_idx)
                classifier = SequenceClassifier(embed_dim, 256, num_classes).to(device)
                train(classifier, train_dataset, 8, 5, 0.001, device)
                acc, prec, recall, f1 = evaluate(classifier, val_dataset, 8, device)
                metrics['acc'].append(acc)
                metrics['prec'].append(prec)
                metrics['recall'].append(recall)
                metrics['f1'].append(f1)

            # write evaluation results for each label
            f.write(f"\nResults for {file_name}:\n")
            for metric, values in metrics.items():
                values_str = ", ".join(f"{v:.4f}" for v in values) 
                f.write(f"{metric.capitalize()} - Values: [{values_str}], Mean: {np.mean(values):.4f}, Std: {np.std(values):.4f}\n")
            f.write("-" * 50 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate model performance.")

    parser.add_argument("-m", "--model_path", required=True, help="Model file path.")
    parser.add_argument("-b", "--is_bert", action='store_true', help="Use BERT model.")
    parser.add_argument("-c", "--corpus_path", required=True, help="Path to the corpus file.")
    parser.add_argument("-ld", "--label_dir", required=True, help="Directory containing label files.")
    parser.add_argument("-o", "--output_dir", required=True, help="Output directory.")
    parser.add_argument("-t", "--tokenizer_path", help="Tokenizer path (required for BERT).")
    parser.add_argument("-l", "--input_label", help="Specific label file name.")

    args = parser.parse_args()

    # if use BERT model, one must specify tokenizer path
    if args.is_bert and not args.tokenizer_path:
        parser.error("--is_bert requires --tokenizer_path.")
    
    # create output dir if not exist
    os.makedirs(args.output_dir, exist_ok=True)

    main_eval_loop(
        model_path=args.model_path,
        is_bert=args.is_bert,
        corpus_path=args.corpus_path,
        label_dir=args.label_dir,
        output_dir=args.output_dir,
        tokenizer_path=args.tokenizer_path,
        input_label=args.input_label
    )

if __name__ == "__main__":
    main()