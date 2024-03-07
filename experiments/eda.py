import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize
from Levenshtein import ratio, distance
from itertools import product

AngTextPath = '../data/AngOrdtext'
AngDatePath = '../data/AngOrddate'
AngIDPath = '../data/AngOrdID'
EngTextPath = '../data/EngOrdtext'
EngDatePath = '../data/EngOrddate'

def read_text_data(filepath):    
    sentences = []
    for line in open(filepath):
        words = line.split()
        sentence = [word for word in words]
        sentences.append(sentence)
    return sentences

def read_data(filepath):
    all_words = []
    for line in open(filepath):
        words = line.split()
        sentence = [word for word in words]
        all_words.extend(sentence)
    return all_words

def get_word_count(sentences):
    all_words = [word for sentence in sentences for word in sentence]
    word_count = len(all_words)
    distinct_word_count = len(set(all_words))
    return all_words, word_count, distinct_word_count

def get_distinct_word(df):
    all_word = []
    for text in df['Text']:
        tokens = text.split()
        all_word.extend(tokens)
    distinct_word = set(all_word)
    return distinct_word

def get_rank_frequency(all_words):
    word_frequency = {}
    for word in all_words:
        if word in word_frequency:
            word_frequency[word] += 1
        else:
            word_frequency[word] = 1
    word_frequency = dict(sorted(word_frequency.items(), key=lambda item: item[1], reverse=True))
    word_frequency = pd.DataFrame(list(word_frequency.items()), columns=['Word', 'Frequency'])
    return word_frequency

def max_similarity(text1, text2):
    """
    return normalized levenshtein similarity between two strings
    """
    return 1 - distance(text1, text2) / max(len(text1), len(text2))

def token_edit_levenstein_similarity_normalized(distinct_word_list):
    '''
    takes a list of distinct words,
    returns a dataframe with all possible combinations of words and their similarity scores
    '''
    df = pd.DataFrame(set(product(distinct_word_list, repeat=2)), columns=['Word1', 'Word2'])
    # ratio = 1 - (distance / (len1 + len2))
    df["ratio_similarity"] = df.apply(lambda x: ratio(x['Word1'], x['Word2']), axis=1)
    # max_similarity = 1 - distance / max(len1, len2)
    df["max_similarity"] = df.apply(lambda x: max_similarity(x['Word1'], x['Word2']), axis=1)
    df = df[df['max_similarity'] != 1.0]
    return df

if __name__ == "__main__":
    # read data
    AngText = read_text_data(AngTextPath)
    AngDate = read_data(AngDatePath)
    EngText = read_text_data(EngTextPath)
    EngDate = read_data(EngDatePath)
    
    # get word count
    AngText_all_words, AngText_word_count, AngText_distinct_word_count = get_word_count(AngText)
    EngText_all_words, EngText_word_count, EngText_distinct_word_count = get_word_count(EngText)
    
    # get rank frequency
    AngText_word_frequency = get_rank_frequency(AngText_all_words)
    EngText_word_frequency = get_rank_frequency(EngText_all_words)
    
    # get similarity scores
    AngText_similarity = token_edit_levenstein_similarity_normalized(AngText_all_words)
    EngText_similarity = token_edit_levenstein_similarity_normalized(EngText_all_words)
    ang_text = read_text_data(AngTextPath)
    ang_date = read_data(AngDatePath)
    ang_id = read_data(AngIDPath)
    ang_df = pd.DataFrame({'Text': ang_text,
                            'Date': ang_date,})
    ang_df['Text'] = ang_df['Text'].apply(lambda x: ' '.join(x))
    ang_df['WordCount'] = ang_df['Text'].apply(lambda x: len(x.split()))
    ang_df['Date'] = ang_df['Date'].astype(int)