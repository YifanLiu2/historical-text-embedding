from nltk.util import ngrams
import torch
from .base_dataset import BaseData

class SubwordData(BaseData):
    """
    A dataset class designed for generating training data for models that incorporate subword information.

    This class processes a text corpus to create training pairs based on the Skip-gram model, enhanced with 
    subword information for each word.

    :param corpus_file: The path to the corpus file.
    :param vocab_size: The maximum size of the vocabulary.
    :param window_size: The size of the context window around each target word.
    :param negative_size: The size of the negative sample per context word.
    :param sample_size: The number of corpus to sample.
    :param subword: The size of the subword n-grams to be generated.

    :raises ValueError: If `subword` is less than or equal to 0.
    :raises ValueError: If `sample_size` is less than or equal to 0.
    :raises ValueError: If `sample)size` is more than corpora size.

    Attributes:
        vocab_size (int): The size of the vocabulary.
        window_size (int): The size of the context window.
        negative_size (int): The size of the negative sample per context word.
        subword (int): The size of the subword n-grams.
        corpora (List): A list of list of tokens from each corpus
        word2id (Dict[str, int]): A dictionary mapping words to their unique IDs.
        id2word (Dict[int, str]): A dictionary mapping IDs back to words.
        sub2id (Dict[str, int]): A dictionary mapping subword n-grams to unique IDs.
        train_data (List[Tuple[int, int, List[int], List[int], int]]): The generated training data.
    """
    def __init__(self, corpus_file, vocab_size, window_size, negative_size, subword, sample_size=None):
        if subword <= 0:
            raise ValueError("subword must be a positive integer")
        self.subword = subword

        super().__init__(corpus_file, vocab_size, window_size, negative_size, sample_size)

        # initialize global vocab
        self.sub2id = {}
        self._gen_subword_vocab()
        
        # process data in corpus level
        for corpus in self.corpora:
            self.train_data.extend(self._gen_train_data(corpus))
    
    def _gen_subword_vocab(self):
        """
        Generate subword mapping for the entire corpora.
        """
        all_subwords = set(subword for corpus in self.corpora for token in corpus for subword in gen_ngrams(token, self.subword))
        self.sub2id = {sub: i for i, sub in enumerate(all_subwords)}
    
    def _gen_train_data(self, corpus):
        """
        Generates the training data from the tokenized corpus, incorporating subword information.

        This method iterates over each token in the corpus, treating it as a target word, and generates context 
        and negative samples within the specified window size. Each sample includes the target and context (or 
        negative) word IDs, along with the IDs of their subword n-grams.

        :param corpus: A list of tokens for the corpus.
        :return: A list of training samples, where each sample is a tuple containing the target ID, context ID, 
        target subword IDs, context subword IDs, and the label (1 for positive samples, 0 for negative samples).
        """
        train_data = []
        for i, target in enumerate(corpus):
            target_id = self.word2id.get(target, None)
            if target_id is None:
                continue
            target_subs = gen_ngrams(target, self.subword)
            target_sub_id = [self.sub2id[sub] for sub in target_subs]
            for j in range(max(0, i - self.window_size), min(len(corpus), i + self.window_size + 1)):
                if i != j:
                    context = corpus[j]
                    context_id = self.word2id.get(context, None)
                    if context_id is None:
                        continue
                    context_subs = gen_ngrams(context, self.subword)
                    context_sub_id = [self.sub2id[sub] for sub in context_subs]
                    train_data.append((target_id, context_id, target_sub_id, context_sub_id, 1))
                    
                    negative_samples = set()
                    while len(negative_samples) < self.negative_size:
                        negative_id = torch.randint(0, self.vocab_size, (1,)).item()
                        if negative_id != target_id and negative_id not in negative_samples:
                            negative_samples.add(negative_id)
                            negative = self.id2word[negative_id]
                            negative_subs = gen_ngrams(negative, self.subword)
                            negative_sub_id = [self.sub2id[sub] for sub in negative_subs]
                            train_data.append((target_id, negative_id, target_sub_id, negative_sub_id, 0)) 

        return train_data
    
    def __getitem__(self, idx):
        target_id, context_id, target_sub, context_sub, label = self.train_data[idx]
        
        return (
            torch.tensor(target_id, dtype=torch.long), 
            torch.tensor(context_id, dtype=torch.long), 
            torch.tensor(target_sub, dtype=torch.long), 
            torch.tensor(context_sub, dtype=torch.long),
            torch.tensor(label, dtype=torch.float)
        )


def gen_ngrams(word, n):
    """
    Generates n-grams for a given word.

    :param word: The word for which to generate n-grams.
    :param n: The size of the n-grams.
    :return: A list of n-grams generated from the word.
    """
    word = f"<{word}>"
    return list(ngrams(word, n))


if __name__ == "__main__":
    file = "data/EngOrdText"
    vocab_size = 1000
    window_size = 5
    negative_size = 5
    subword = 4
    sample_size = 500

    dataset = SubwordData(file, vocab_size, window_size, negative_size, subword, sample_size)
    dataset.save_dataset("outputs/test_norman_subword_data")
