import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from Levenshtein import ratio, distance
from itertools import product

AngTextPath = 'data/AngOrdtext'
AngDatePath = 'data/AngOrddate'
EngTextPath = 'data/EngOrdtext'
EngDatePath = 'data/EngOrddate'

"""AngTextPath = 'data\AngOrdtext'
AngDatePath = 'data\AngOrddate'
EngTextPath = 'data\EngOrdtext'
EngDatePath = 'data\EngOrddate'"""

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
    all_words = [word for word in sentences]
    word_count = len(all_words)
    distinct_word_count = len(set(all_words))
    return all_words, word_count, distinct_word_count

def get_word(df):
    all_word = []
    for text in df['Text']:
        tokens = text.split()
        all_word.extend(tokens)
    distinct_word = set(all_word)
    return all_word, distinct_word

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

def similarity_report(distinct_word_list, threshold=0.85, desc='Calculating similarity'):
    '''
    takes a list of distinct words,
    returns a dataframe with combinations with similarity score higher than 0.85
    '''
    pbar = tqdm(total=len(distinct_word_list), desc=desc)
    dfs = []
    for word1 in distinct_word_list:
        for word2 in distinct_word_list:
            similarity = max_similarity(word1, word2)
            if 1 > similarity > threshold:
                df = pd.DataFrame({'Word1': [word1], 'Word2': [word2], 'Similarity': [similarity]})
                dfs.append(df)
        pbar.update(1)
    pbar.close()
    df = pd.concat(dfs, ignore_index=True)
    return df

def similarity_summary(df, desc = 'Grouping similar words'):
    '''
    takes a dataframe with word1 and word2 that are similar,
    returns a dataframe with groups of similar words
    '''
    groups = []
    explored = set()
    pbar = tqdm(total=len(df['Word1']), desc=desc)
    for word1 in df['Word1']:
        if word1 not in explored:
            group = [word1]
            explored.add(word1)
            word2 = df[df['Word1'] == word1]['Word2']
            group.extend(word2)
            for w in word2:
                explored.add(w)
            groups.append(group)
        pbar.update(1)
    pbar.close()
    summary_df = pd.DataFrame({'Group': range(1, len(groups)+1), 'Words': [' '.join(group) for group in groups]})
    return summary_df

def word_to_similar_group(word_list, threshold=0.85, desc='Grouping similar words', path='similarity_summary.txt'):
    df_test = similarity_report(word_list, threshold=threshold, desc=desc+ ' report')
    summary_test = similarity_summary(df_test, desc=desc + ' summary')
    summary_test.to_csv(path, sep='\t', index=False)

if __name__ == "__main__":
    # read data
    AngText = read_text_data(AngTextPath)
    AngDate = read_data(AngDatePath)
    EngText = read_text_data(EngTextPath)
    EngDate = read_data(EngDatePath)
    
    # get data frame
    ang_df = pd.DataFrame({'Text': AngText, 'Date': AngDate})
    eng_df = pd.DataFrame({'Text': EngText, 'Date': EngDate})

    # convert date type
    ang_df['Date'] = ang_df['Date'].astype(int)
    eng_df['Date'] = eng_df['Date'].astype(int)
    ang_df['Text'] = ang_df['Text'].apply(lambda x: ' '.join(x))
    eng_df['Text'] = eng_df['Text'].apply(lambda x: ' '.join(x))
    ang_df['Text'] = ang_df['Text'].astype(str)
    eng_df['Text'] = eng_df['Text'].astype(str)


    # filter for specific time periods
    ang_spe_df = ang_df[(ang_df['Date'] >= 900) & (ang_df['Date'] <= 1066)]
    eng_spe_df = eng_df[(eng_df['Date'] >= 1067) & (eng_df['Date'] <= 1198)]

    ang_spec_all_words, ang_spec_words_set = get_word(ang_spe_df)
    eng_spec_all_words, eng_spec_words_set = get_word(eng_spe_df)
    
    ang_spe_words_list = list(ang_spec_words_set)
    eng_spe_words_list = list(eng_spec_words_set)

    word_to_similar_group(ang_spe_words_list, 
                          desc='Grouping similar words in anglo-saxon', 
                          path='experiments/exp_result/anglo_saxon_similarity_summary.txt')
    
    word_to_similar_group(eng_spe_words_list,
                        desc='Grouping similar words in middle english', 
                        path='experiments/exp_result/middle_english_similarity_summary.txt')
