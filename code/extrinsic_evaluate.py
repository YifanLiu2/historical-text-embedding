from sklearn.metrics import precision_score, recall_score, accuracy_score
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from gensim.models import KeyedVectors
from torch.nn.utils.rnn import pad_sequence
import os


def get_absolute_file_paths(folder_path):
    """Extracts and returns absolute file paths."""
    return [
        os.path.join(folder_path, file)
        for file in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, file))
    ]


def read_file(file_path):
    """Reads text data from a file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.readlines()


def read_labels(file_path):
    """Reads labels from a file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return [int(label) for line in file for label in line.strip().split()]


def sentence_to_embedding(sentence, embeddings):
    """Converts a sentence to an embedding."""
    word_embeddings = [
        embeddings[word] for word in sentence.split() if word in embeddings
    ]
    if not word_embeddings:
        embedding_dim = embeddings.vector_size
        return torch.zeros(1, embedding_dim)
    return torch.tensor(np.stack(word_embeddings), dtype=torch.float32)


def pad_embedding_sentences(sentences, embeddings):
    """Pads embedded sentences."""
    embedding_sentences = [
        sentence_to_embedding(sentence, embeddings) for sentence in sentences
    ]
    return pad_sequence(embedding_sentences, batch_first=True, padding_value=0).reshape(
        len(sentences), -1
    )


def train_MLP(dataset, model, batch_size, epoch_num, learning_rate):
    """Trains a simple MLP model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epoch_num):
        total_loss = 0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")


def evaluate_model_performance(model, dataset, batch_size):
    """Evaluates model performance."""
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_predictions, all_true_labels = [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_true_labels, all_predictions)
    precision = precision_score(all_true_labels, all_predictions, average="binary")
    recall = recall_score(all_true_labels, all_predictions, average="binary")
    return accuracy, precision, recall


def eval_embeddings(
    model_path,
    text_path,
    labels_path,
    learning_rate=0.001,
    train_ratio=0.7,
    batch_size=10,
    epoch_num=2,
):
    """Evaluates embeddings and model."""
    results_df = pd.DataFrame(columns=["FileName", "Accuracy", "Precision", "Recall"])
    embedding_matrix = KeyedVectors.load_word2vec_format(model_path, binary=False)
    sentences = read_file(text_path)
    sentences_padded = pad_embedding_sentences(sentences, embedding_matrix)
    N, input_size = sentences_padded.shape

    label_lst = get_absolute_file_paths(labels_path)
    for label_path in label_lst:
        file_name = os.path.basename(label_path).split(".")[0]
        labels = torch.tensor(read_labels(label_path), dtype=torch.long)
        output_size = len(torch.unique(labels))
        model = nn.Linear(input_size, output_size)

        dataset = TensorDataset(sentences_padded, labels)
        train_size = int(train_ratio * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size]
        )

        train_MLP(train_dataset, model, batch_size, epoch_num, learning_rate)

        train_performance = evaluate_model_performance(model, train_dataset, batch_size)
        test_performance = evaluate_model_performance(model, test_dataset, batch_size)

        for suffix, performance in zip(
            ["_train", "_test"], [train_performance, test_performance]
        ):
            results_df = results_df.append(
                {
                    "FileName": file_name + suffix,
                    "Accuracy": performance[0],
                    "Precision": performance[1],
                    "Recall": performance[2],
                },
                ignore_index=True,
            )

    return results_df


# Example usage, adjust paths as necessary
# model_path = "path/to/your/model"
# text_path = "path/to/your/texts"
# labels_path = "path/to/your/labels"
# results_df = eval_embeddings(model_path, text_path, labels_path)
# print(results_df)

if __name__ == "__main__":
    pd_eval = pd.DataFrame()
    model_path = "trained_model/eng_model.bin"
    text_path = "../data/EngOrdtext"
    labels_path = "../data/eval_data"

    results_df = eval_embeddings(model_path, text_path, labels_path)
    print(results_df)
