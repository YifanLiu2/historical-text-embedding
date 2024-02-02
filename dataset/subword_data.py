from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from collections import Counter
from torch.utils.data import Dataset
import torch
import pickle

class SubwordData(Dataset):
    """
    A dataset class designed for generating training data for models that incorporate subword information.

    This class processes a text corpus to create training pairs based on the Skip-gram model, enhanced with 
    subword information for each word.

    :param corpus_file: The path to the corpus file.
    :param vocab_size: The maximum size of the vocabulary.
    :param window_size: The size of the context window around each target word.
    :param subword: The size of the subword n-grams to be generated.

    :raises ValueError: If `subword` is less than or equal to 0.

    Attributes:
        vocab_size (int): The size of the vocabulary.
        window_size (int): The size of the context window.
        subword (int): The size of the subword n-grams.
        tokens (List[str]): A list of tokens obtained from the corpus.
        word2id (Dict[str, int]): A dictionary mapping words to their unique IDs.
        id2word (Dict[int, str]): A dictionary mapping IDs back to words.
        sub2id (Dict[str, int]): A dictionary mapping subword n-grams to unique IDs.
        train_data (List[Tuple[int, int, List[int], List[int], int]]): The generated training data.
    """
    def __init__(self, corpus_file, vocab_size, window_size, subword):
        self.vocab_size = vocab_size
        self.window_size = window_size

        with open(corpus_file, "r") as f:
            text = f.read().lower()

        if subword <= 0:
            raise ValueError("subword must be a positive integer")
        self.subword = subword

        self.tokens = word_tokenize(text)
        word_count = Counter(self.tokens)
        self.word2id = {word: i for i, (word, _) in enumerate(word_count.most_common(vocab_size))}
        self.id2word = {i: word for word, i in self.word2id.items()} 
        
        subwords = set([subword for token in self.tokens for subword in gen_ngrams(token, self.subword)])
        self.sub2id = {sub : i for i, sub in enumerate(subwords)}
        self.train_data = self.gen_train_data()
    
    def gen_train_data(self):
        """
        Generates the training data from the tokenized corpus, incorporating subword information.

        This method iterates over each token in the corpus, treating it as a target word, and generates context 
        and negative samples within the specified window size. Each sample includes the target and context (or 
        negative) word IDs, along with the IDs of their subword n-grams.

        :return: A list of training samples, where each sample is a tuple containing the target ID, context ID, 
        target subword IDs, context subword IDs, and the label (1 for positive samples, 0 for negative samples).
        """
        train_data = []
        for i, target in enumerate(self.tokens):
            target_id = self.word2id.get(target, None)
            if target_id is None:
                continue
            target_subs = gen_ngrams(target, self.subword)
            target_sub_id = [self.sub2id[sub] for sub in target_subs]
            for j in range(max(0, i - self.window_size), min(len(self.tokens), i + self.window_size)):
                if i != j:
                    context = self.tokens[j]
                    context_id = self.word2id.get(context, None)
                    if context_id is None:
                        continue
                    context_subs = gen_ngrams(context, self.subword)
                    context_sub_id = [self.sub2id[sub] for sub in context_subs]
                    train_data.append((target_id, context_id, target_sub_id, context_sub_id, 1))
                    
                    for _ in range(self.window_size):
                        find = False
                        while not find:
                            negative_id = torch.randint(0, self.vocab_size, (1,)).item()
                            if negative_id != target_id:
                                find = True
                        negative = self.id2word[negative_id]
                        negative_subs = gen_ngrams(negative, self.subword)
                        negative_sub_id = torch.tensor([self.sub2id[sub] for sub in negative_subs], dtype=torch.long)
                        train_data.append((target_id, negative_id, target_sub_id, negative_sub_id, 0))
        return train_data
    
    def __len__(self):
        return len(self.train_data)
    
    def __getitem__(self, idx):
        target_id, context_id, target_sub, context_sub, label = self.train_data[idx]
        
        return (
            torch.tensor(target_id, dtype=torch.long), 
            torch.tensor(context_id, dtype=torch.long), 
            torch.tensor(target_sub, dtype=torch.long), 
            torch.tensor(context_sub, dtype=torch.long),
            torch.tensor(label, dtype=torch.float)
        )
    
    def save_dataset(self, save_path):
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load_dataset(cls, load_path):
        with open(load_path, 'rb') as f:
            dataset = pickle.load(f)
        return dataset


def gen_ngrams(word, n):
    """
    Generates n-grams for a given word.

    :param word: The word for which to generate n-grams.
    :param n: The size of the n-grams.
    :return: A list of n-grams generated from the word.
    """
    word = f"<{word}>"
    return list(ngrams(word, n))