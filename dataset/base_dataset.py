from nltk.tokenize import word_tokenize
from collections import Counter
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
import pickle
import random

class BaseData(Dataset, ABC):
    """
    A base dataset class adaptable to various word embedding model.

    :param corpus_file: The path to the corpus file.
    :param vocab_size: The maximum size of the vocabulary.
    :param window_size: The size of the context window around each target word.
    :param negative_size: The size of the negative sample per context word.
    :param sample_size: The number of corpus to sample.
    :raises ValueError: If `sample_size` is less than or equal to 0.
    :raises ValueError: If `sample)size` is more than corpora size.

    Attributes:
        vocab_size (int): The size of the vocabulary.
        window_size (int): The size of the context window.
        negative_size (int): The size of the negative sample per context word.
        corpora (List): A list of list of tokens for each corpus
        word2id (Dict[str, int]): A dictionary mapping words to their unique IDs.
        id2word (Dict[int, str]): A dictionary mapping IDs back to words.
        train_data (List[Tuple[int, int, List[int], List[int], int]]): The generated training data.
    """

    def __init__(self, corpus_file, vocab_size, window_size, negative_size, sample_size=None):
        self.vocab_size = vocab_size
        self.window_size = window_size
        self.negative_size = negative_size

        # initialize global vocab
        self.corpora = [] 
        self.word2id = {}
        self.id2word = {}
        self.sub2id = {}
        self.train_data = []
        
        # read corpus
        corpora = []
        with open(corpus_file, "r") as f:
            for line in f:
                line = line.lower()
                tokens = word_tokenize(line)
                corpora.append(tokens)
        
        # sampling if needed
        if sample_size is not None:
            if sample_size > len(corpora):
                raise ValueError("sample_size must be smaller than corpora size")
            if sample_size <= 0:
                raise ValueError("sample size must be a positive integer")
            corpora = random.sample(corpora, sample_size)
        self.corpora = corpora

        # set gloabl vocab
        self._gen_global_vocab()
    
    def _gen_global_vocab(self):
        """
        Generates a global vocabulary for the entire dataset.
        """
        tokens = [token for corpus in self.corpora for token in corpus]
        word_count = Counter(tokens)
        self.word2id = {word: i for i, (word, _) in enumerate(word_count.most_common(self.vocab_size))}
        self.id2word = {i: word for word, i in self.word2id.items()}
    
    @abstractmethod
    def _gen_train_data(self, corpus):
        """
        Generates the training data from the tokenized corpus.

        :param corpus: A list of tokens for the corpus.
        :return: A list of training samples
        """
        raise NotImplementedError
    
    def __len__(self):
        return len(self.train_data)
    
    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError
    
    def save_dataset(self, save_path):
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load_dataset(cls, load_path):
        with open(load_path, 'rb') as f:
            dataset = pickle.load(f)
        return dataset