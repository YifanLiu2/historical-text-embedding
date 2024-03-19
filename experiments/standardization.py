import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from Levenshtein import ratio, distance
from itertools import product
from collections import Counter

AngTextPath = 'data/AngOrdtext'
EngTextPath = 'data/EngOrdtext'
FrTextPath = 'data/FrOrdtext'

def read_words(filepath):
    '''
    return a list of words
    each word is a string
    '''    
    text = []
    for line in open(filepath):
        words = line.split()
        sentence = [word for word in words]
        text.extend(sentence)
    word_map2 = {'æ': 'e', 'ae': 'e', 'ð': 'th', 'þ': 'th'}
    for old, new in word_map2.items():
        text = [word.replace(old, new) for word in text]
    return text

def read_sentences(filepath):  
    '''
    return a list of sentences
    each sentence is a string
    ''' 
    sentences = []
    for line in open(filepath):
        words = line.split()
        sentence = ' '.join(words)
        word_map2 = {'æ': 'e', 'ae': 'e', 'ð': 'th', 'þ': 'th'}
        for old, new in word_map2.items():
            sentence = sentence.replace(old, new)
        sentences.append(sentence)
    return sentences

def get_rank_frequency(all_words):
    word_frequency = Counter(all_words)
    word_frequency = dict(sorted(word_frequency.items(), key=lambda item: item[1], reverse=True))
    word_frequency = pd.DataFrame(list(word_frequency.items()), columns=['Word', 'Frequency'])
    return word_frequency

def max_similarity(text1, text2):
    """
    return normalized levenshtein similarity between two strings
    """
    if type(text1) != str:
        text1 = str(text1)
    if type(text2) != str:
        text2 = str(text2)
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
    if len(dfs) == 0:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    return df

def similarity_summary(df, desc = 'Grouping similar words'):
    '''
    takes a dataframe with word1 and word2 that are similar,
    returns a dataframe with groups of similar words
    '''
    if df.empty:
        return pd.DataFrame()
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

def summary_to_dict(all_words_frequency, summary_df):
    '''
    from a group of words whose similarity score is higher than 0.9 and frequency is higher than 5,
    we keep the word with highest frequency
    '''
    summary = summary_df['Words'].str.split(' ').tolist()
    summary_dict = {}
    pbar = tqdm(total=len(summary))
    for group in summary:
        max_word = ''
        max_freq = -1
        for word in group:
            freq = all_words_frequency[all_words_frequency['Word'] == word]['Frequency'].values[0]
            if freq > max_freq:
                max_freq = freq
                max_word = word
        for word in group:
            if word != max_word:
                summary_dict[word] = max_word
        pbar.update(1)
    pbar.close()
    return summary_dict

def standardize_p1():
    '''
    get similarity summaries for two thresholds, 0.85 and 0.90
    '''
    ang_text = read_words(AngTextPath)
    eng_text = read_words(EngTextPath)
    fr_text = read_words(FrTextPath)
    all_words = ang_text + eng_text + fr_text
    word_frequency = get_rank_frequency(all_words)
    word_frequency.to_csv('experiments/result_2/word_frequency.txt', index=False)
    distinct_word_list = word_frequency['Word']
    similarity_df_085 = similarity_report(distinct_word_list, threshold=0.85)
    similarity_df_085.to_csv('experiments/result_2/similarity_df_085.txt', index=False)
    similarity_df_090 = similarity_df_085[similarity_df_085['Similarity'] > 0.90]
    similarity_summary_df_085 = similarity_summary(similarity_df_085)
    similarity_summary_df_090 = similarity_summary(similarity_df_090)
    similarity_summary_df_085.to_csv('experiments/result_2/similarity_summary_df_085.txt', index=False)
    similarity_summary_df_090.to_csv('experiments/result_2/similarity_summary_df_090.txt', index=False)


def standardize_p2():
    '''
    combine the two similarity summary dataframes
    for scarce words with frequency not greater than 5, 
        we keep the word with highest frequency according to 0.85 threshold
    for words with frequency greater than 5,
        we keep the word with highest frequency according to 0.90 threshold
    '''
    word_frequency = pd.read_csv('experiments/result_2/word_frequency.txt')
    similarity_summary_df_085 = pd.read_csv('experiments/result_2/similarity_summary_df_085.txt')
    similarity_summary_df_090 = pd.read_csv('experiments/result_2/similarity_summary_df_090.txt')
    summary_dict_085 = summary_to_dict(word_frequency, similarity_summary_df_085)
    summary_dict_090 = summary_to_dict(word_frequency, similarity_summary_df_090)
    summary_dict_085 = pd.DataFrame(list(summary_dict_085.items()), columns=['Word', 'Replacement'])
    summary_dict_085.to_csv('experiments/result_2/summary_dict_085.txt', index=False)
    summary_dict_090 = pd.DataFrame(list(summary_dict_090.items()), columns=['Word', 'Replacement'])
    summary_dict_090.to_csv('experiments/result_2/summary_dict_090.txt', index=False)

def standardize_p3():
    '''
    '''
    word_frequency = pd.read_csv('experiments/result_2/word_frequency.txt')
    summary_dict_085 = pd.read_csv('experiments/result_2/summary_dict_085.txt')
    summary_dict_090 = pd.read_csv('experiments/result_2/summary_dict_090.txt')
    summary_dict_combined = {}
    pb = tqdm(total=len(word_frequency['Word']))
    for word in word_frequency['Word']:
        temp = word_frequency[word_frequency['Word'] == word]['Frequency'].values
        if len(temp) == 0:
            continue
        freq = temp[0]
        if freq > 5:
            replacement = summary_dict_090[summary_dict_090['Word'] == word]['Replacement'].values
            if len(replacement) > 0:
                summary_dict_combined[word] = replacement[0]
        else:
            replacement = summary_dict_085[summary_dict_085['Word'] == word]['Replacement'].values
            if len(replacement) > 0:
                summary_dict_combined[word] = replacement[0]
        pb.update(1)
    pb.close()
    summary_dict_combined = pd.DataFrame(list(summary_dict_combined.items()), columns=['Word', 'Replacement'])
    summary_dict_combined.to_csv('experiments/result_2/summary_dict_combined.txt', index=False)

def standardize_p4_helper(textpath, summary_dict_combined):
    text = read_sentences(textpath)
    pbar = tqdm(total=len(text))
    for i, sentence in enumerate(text):
        for old, new in summary_dict_combined.items():
            sentence = sentence.replace(old, new)
        text[i] = sentence
        pbar.update(1)
    pbar.close()
    return pd.DataFrame(text)

def standardize_p4():
    stand_path = 'experiments/standardized_data_d5'
    summary_dict_combined = pd.read_csv('experiments/result_2/summary_dict_combined.txt')
    summary_dict_combined = dict(zip(summary_dict_combined['Word'], summary_dict_combined['Replacement']))

    ang_text_df = standardize_p4_helper(AngTextPath, summary_dict_combined)
    eng_text_df = standardize_p4_helper(EngTextPath, summary_dict_combined)
    fr_text_df = standardize_p4_helper(FrTextPath, summary_dict_combined)
    print('Merging')
    all_text = pd.concat([ang_text_df, eng_text_df, fr_text_df], ignore_index=True)
    print('Saving')
    all_text.to_csv(stand_path + '/AllStandText', index=False, header=False)
    ang_text_df.to_csv(stand_path + '/AngStandText', index=False, header=False)
    eng_text_df.to_csv(stand_path + '/EngStandText', index=False, header=False)
    fr_text_df.to_csv(stand_path + '/FrStandText', index=False, header=False)


def standardize_evaluation():
    all_text = read_words('experiments/standardized_data_d5/AllStandText')
    all_rf = get_rank_frequency(all_text)
    print(all_rf.head(10))
    print(len(all_rf))
    

if __name__ == '__main__':
    standardize_evaluation()