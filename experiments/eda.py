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
FrTextPath = 'data/FrOrdtext'
FrDatePath = 'data/FrOrddate'

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

def get_word(texts):
    all_word = []
    for text in texts:
        all_word.extend(text)
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

def standardize_letters(sentence):
    # replace other letters
    word_map2 = {'æ': 'e', 'ae': 'e', 'ð': 'th', 'þ': 'th'}
    new_sentence = []
    for word in sentence:
        new_word = word
        for letter in new_word: 
            if letter in word_map2:
                new_word = new_word.replace(letter, word_map2[letter])
        new_sentence.append(new_word)
    return new_sentence 

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

def normalize_scarce_words(scarce_words, word_list, threshold=0.85, desc='Normalizing scarce words'):
    '''
    for each word in scarce_words, find the most similar word in word_list
    return a dictionary with the most similar word for each word in scarce_words
    '''
    scarce_words_dict = {}
    pbar = tqdm(total=len(scarce_words), desc='Normalizing scarce words')
    for word in scarce_words:
        max_s = 0
        max_word = ''
        max_freq = -1
        for w in word_list:
            similarity = max_similarity(word, w)
            if 1 > similarity > max_s:
                max_s = similarity
                max_word = w
        if max_s > threshold:
            scarce_words_dict[word] = max_word
        pbar.update(1)
    pbar.close()
    return scarce_words_dict

def compute_similarity(word, possible_words, threshold, scarce_words_dict):
    max_s = 0
    max_word = ''
    
    # Compute similarities using vectorized operations
    similarities = [max_similarity(word, w) for w in possible_words]
    
    max_sim_idx = np.argmax(similarities)
    max_s = similarities[max_sim_idx]
    max_word = possible_words.iloc[max_sim_idx]
    
    # Check similarity threshold
    if max_s > threshold:
        scarce_words_dict[word] = max_word

def scarce_words_to_dict(df_freq, threshold=0.85, desc='Normalizing scarce words'):
    '''
    takes a dataframe with word and frequency, and a threshold,
    returns a dictionary with the most similar word for each word in scarce_words
    '''
    scarce_words_dict = {}
    len_scarce = df_freq[df_freq['Frequency'] <= 5].shape[0]

    pbar = tqdm(total=len_scarce, desc='Normalizing scarce words')
    for i in range(1, 6):
        scarce_words = df_freq[df_freq['Frequency'] == i]['Word']
        possible_words = df_freq[df_freq['Frequency'] > i]['Word']
        for word in scarce_words:
            compute_similarity(word, possible_words, threshold, scarce_words_dict)
            pbar.update(1)
    pbar.close()
    return scarce_words_dict


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

def word_to_similar_group(word_list, threshold=0.85, threshold_s = 0.9, desc='Grouping similar words', 
                          path1='similarity_report.txt', 
                          path2='similarity_summary.txt'):
    df_test = similarity_report(word_list, threshold=threshold, desc=desc+ ' report')
    df_test.to_csv(path1, sep='\t', index=False)
    df_test = df_test[df_test['Similarity'] > threshold_s]
    summary_test = similarity_summary(df_test, desc=desc + ' summary')
    summary_test.to_csv(path2, sep='\t', index=False)

def file_to_dict(filepath):
    dictionary = {}

    with open(filepath, 'r') as file:
        next(file)
        for line in file:
            key, value = line.strip().split()[0], line.strip().split()[1]
            dictionary[key] = value

    return dictionary

def file_to_scarce_dict(filepath):
    dictionary = {}

    with open(filepath, 'r') as file:
        next(file)
        for line in file:
            key, value = line.strip().split()[1], line.strip().split()[2]
            dictionary[key] = value

    return dictionary


def main_part_1():
    '''
    Generate txt file for word similarity report in 0.85 generation threshold
    '''
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

    ang_spec_all_words, ang_spec_words_set = get_word(ang_spe_df["Text"])
    eng_spec_all_words, eng_spec_words_set = get_word(eng_spe_df["Text"])
    all_words_set = ang_spec_words_set.union(eng_spec_words_set)
    
    ang_spe_words_list = list(ang_spec_words_set)
    eng_spe_words_list = list(eng_spec_words_set)
    all_words_list = list(all_words_set)
    
    df_test = similarity_report(ang_spe_words_list, threshold=0.85, desc='Grouping similar words in all texts report').sort_values(by='Similarity', ascending=False)
    df_test.to_csv('experiments/exp_result/anglo_saxon_similarity_report_0.85.txt', sep='\t', index=False)
    df_test = similarity_report(eng_spe_words_list, threshold=0.85, desc='Grouping similar words in all texts report').sort_values(by='Similarity', ascending=False)
    df_test.to_csv('experiments/exp_result/middle_english_similarity_report_0.85.txt', sep='\t', index=False)
    df_test = similarity_report(all_words_list, threshold=0.85, desc='Grouping similar words in all texts report').sort_values(by='Similarity', ascending=False)
    df_test.to_csv('experiments/exp_result/all_texts_similarity_report_0.85.txt', sep='\t', index=False)

def main_part_2():
    summary_threshold = 0.88
    df_similar = pd.read_csv('experiments/exp_result/all_texts_similarity_report_0.85.txt', sep='\t')
    df_similar = df_similar[df_similar['Similarity'] > summary_threshold]
    df_summary = similarity_summary(df_similar)
    df_summary.to_csv('experiments/exp_result/all_texts_similarity_summary_{}.txt'.format(summary_threshold), sep='\t', index=False)

def wrapper_p1(Text):
    Text = [standardize_letters(sentence) for sentence in Text]
    all_words,_ = get_word(Text)
    word_frequency = get_rank_frequency(all_words)
    return all_words, word_frequency

def standardize_helper(sentences, scarce_dict, summary_dict, desc='Standardizing texts'):
    stand_sentences = []
    pbar = tqdm(total=len(sentences), desc=desc)
    for sentence in sentences:
        stand_sentence = []
        sentence = standardize_letters(sentence)
        for word in sentence:
            if word in scarce_dict:
                stand_sentence.append(scarce_dict[word])
            elif word in summary_dict:
                if word == "tilesberia":
                    print(word, word["tilesberia"])
                stand_sentence.append(summary_dict[word])
            else:
                stand_sentence.append(word)
        stand_sentences.append(stand_sentence)
        pbar.update(1)
    pbar.close()
    return stand_sentences



def standardization_p1():
    '''
    get word frequency for all texts
    normalize letters
    '''
    AngText = read_text_data(AngTextPath)
    EngText = read_text_data(EngTextPath)
    FrText = read_text_data(FrTextPath)
    AllText = AngText + EngText + FrText

    all_words, all_words_frequency = wrapper_p1(AllText)

    all_words_frequency.to_csv('experiments/result/all_text_word_frequency.txt', sep='\t', index=False)

def standardization_p2():
    '''
    get standardization for scarce words in all texts
    in which scarce words are words has frequency less than or equal to 5
    '''
    all_words_frequency = pd.read_csv('experiments/result/all_text_word_frequency.txt', sep='\t')
    all_words_list = all_words_frequency['Word']        
    scarce_words = all_words_frequency[all_words_frequency['Frequency'] <= 5]['Word']
    rest_words = all_words_frequency[all_words_frequency['Frequency'] > 5]['Word']
    scarce_words_dict = normalize_scarce_words(scarce_words, rest_words)
    scarce_words_df = pd.DataFrame(scarce_words_dict, index=['Value'])  
    scarce_words_df = scarce_words_df.T.reset_index()
    scarce_words_df.to_csv('experiments/result/scarce_words_dict.txt', sep='\t', index=True,
                                           header=None)
    
def standardization_p2_revised():
    '''
    get standardization for scarce words in all texts
    in which scarce words are words has frequency less than or equal to 5
    '''
    all_words_frequency = pd.read_csv('experiments/result/all_text_word_frequency.txt', sep='\t')
    # all_words_frequency = all_words_frequency.sample(frac=0.05)
    scarce_words_dict = scarce_words_to_dict(all_words_frequency, threshold=0.85, desc='Normalizing scarce words')
    scarce_words_df = pd.DataFrame(scarce_words_dict, index=['Value'])  
    scarce_words_df = scarce_words_df.T.reset_index()
    scarce_words_df.to_csv('experiments/result/scarce_words_dict_revised.txt', sep='\t', index=True,
                                           header=None)

def standardization_p3():
    '''
    get standardization for not scarce words in all texts
    in which we keep the words in a group of similar words with threshold 0.9
    '''
    all_words_frequency = pd.read_csv('experiments/result/all_text_word_frequency.txt', sep='\t')
    rest_words = all_words_frequency[all_words_frequency['Frequency'] > 5]['Word']
    word_to_similar_group(rest_words, threshold=0.85, desc='Grouping similar words in all texts',
                          path1='experiments/result/similarity_report_0.85.txt',
                          path2='experiments/result/similarity_summary_0.85.txt')
    
def standardization_p35():
    path1 = 'experiments/result/similarity_report_0.85.txt'
    path2 = 'experiments/result/similarity_summary_0.9.txt'
    df_test = pd.read_csv(path1, sep='\t')
    df_test = df_test[df_test['Similarity'] >= 0.9]
    summary_test = similarity_summary(df_test, desc='summary')
    summary_test.to_csv(path2, sep='\t', index=False)

def standardization_p4():
    '''
    from a group of words whose similarity score is higher than 0.9 and frequency is higher than 5,
    we keep the word with highest frequency
    '''
    all_words_frequency = pd.read_csv('experiments/result/all_text_word_frequency.txt', sep='\t')
    summary = pd.read_csv('experiments/result/similarity_summary_0.9.txt', sep='\t')
    summary = summary['Words'].str.split(' ').tolist()
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
    pd.DataFrame(summary_dict, index=[0]).T.to_csv('experiments/result/summary_dict.txt', sep='\t', index=True,
    header = [""])

def standardization_p5():
    '''
    Given a dictionary of word standardization, we standardize all texts
    '''
    stand_path = 'experiments/standardized_data_d3'
    summary_dict = file_to_dict('experiments/result/summary_dict.txt')
    scarce_dict = file_to_scarce_dict('experiments/result/scarce_words_dict_revised.txt')
    AngText = read_text_data(AngTextPath)
    EngText = read_text_data(EngTextPath)
    FrText = read_text_data(FrTextPath)
    AngStandText = standardize_helper(AngText, scarce_dict, summary_dict, desc='Standardizing AngText')
    EngStandText = standardize_helper(EngText, scarce_dict, summary_dict, desc='Standardizing EngText')
    FrStandText = standardize_helper(FrText, scarce_dict, summary_dict, desc='Standardizing FrText')
    pd.DataFrame(AngStandText).to_csv(stand_path+'/AngStandText', sep=' ', index=False, header=False)
    pd.DataFrame(EngStandText).to_csv(stand_path+'/EngStandText', sep=' ', index=False, header=False)
    pd.DataFrame(FrStandText).to_csv(stand_path+'/FrStandText', sep=' ', index=False, header=False)
    AllStandText = AngStandText + EngStandText + FrStandText
    pd.DataFrame(AllStandText).to_csv(stand_path+'/AllStandText', sep=' ', index=False, header=False)

def standardization_p6():
    '''
    Given a dictionary of word standardization, we standardize all texts
    '''
    stand_path = 'experiments/standardized_data_d3'
    summary_dict = file_to_dict('experiments/result/summary_dict.txt')
    scarce_dict = file_to_scarce_dict('experiments/result/scarce_words_dict_revised.txt')
    AngTextPath = 'experiments/standardized_data_d3/AngStandText'
    EngTextPath = 'experiments/standardized_data_d3/EngStandText'
    FrTextPath = 'experiments/standardized_data_d3/FrStandText'
    AngText = read_text_data(AngTextPath)
    EngText = read_text_data(EngTextPath)
    FrText = read_text_data(FrTextPath)
    AngStandText = standardize_helper(AngText, scarce_dict, summary_dict, desc='Standardizing AngText')
    EngStandText = standardize_helper(EngText, scarce_dict, summary_dict, desc='Standardizing EngText')
    FrStandText = standardize_helper(FrText, scarce_dict, summary_dict, desc='Standardizing FrText')
    pd.DataFrame(AngStandText).to_csv(stand_path+'/AngStandText', sep=' ', index=False, header=False)
    pd.DataFrame(EngStandText).to_csv(stand_path+'/EngStandText', sep=' ', index=False, header=False)
    pd.DataFrame(FrStandText).to_csv(stand_path+'/FrStandText', sep=' ', index=False, header=False)
    print("good")
    AllStandText = AngStandText + EngStandText + FrStandText
    pd.DataFrame(AllStandText).to_csv(stand_path+'/AllStandText', sep=' ', index=False, header=False)
    print("good")

def standardization_evaluation():
    EngDatePath = 'data/EngOrddate'
    FrDatePath = 'data/FrOrddate'
    AngDatePath = 'data/AngOrddate'

    AngTextPath = 'data/AngOrdtext'
    EngTextPath = 'data/EngOrdtext'
    FrTextPath = 'data/FrOrdtext'

    AngStandTextPath = 'experiments/standardized_data_d3/AngStandText'
    EngStandTextPath = 'experiments/standardized_data_d3/EngStandText'
    FrStandTextPath = 'experiments/standardized_data_d3/FrStandText'
    AllStandTextPath = 'experiments/standardized_data_d3/AllStandText'



    AngStandText = read_text_data(AngStandTextPath)
    EngStandText = read_text_data(EngStandTextPath)
    FrStandText = read_text_data(FrStandTextPath)
    AngText = read_text_data(AngTextPath)
    EngText = read_text_data(EngTextPath)
    FrText = read_text_data(FrTextPath)
    AngDate = read_data(AngDatePath)
    EngDate = read_data(EngDatePath)
    FrDate = read_data(FrDatePath)

    ang_df = pd.DataFrame({'Text': AngStandText,
                    'Date': AngDate,
                    'Origin': AngText})

    ang_df['Text'] = ang_df['Text'].apply(lambda x: ' '.join(x))
    ang_df['Origin'] = ang_df['Origin'].apply(lambda x: ' '.join(x))

    ang_ori_all_words = []
    for text in ang_df['Origin']:
        tokens = text.split()
        ang_ori_all_words.extend(tokens)

    ang_stand_all_words = []
    for text in ang_df['Text']:
        tokens = text.split()
        ang_stand_all_words.extend(tokens)

    print("Ang ori", ang_ori_all_words[11])
    print("Ang stand", ang_stand_all_words[11])
    print("Ang ori", len(ang_ori_all_words))
    print("Ang stand", len(ang_stand_all_words))
    print("Ang ori distinct words", len(set(ang_ori_all_words)))
    print("Ang stand distinct words", len(set(ang_stand_all_words)))
    print("different words", len(set(ang_ori_all_words)) - len(set(ang_stand_all_words)))

    eng_df = pd.DataFrame({'Text': EngStandText, 'Origin': EngText})
    fr_df = pd.DataFrame({'Text': FrStandText, 'Origin': FrText})
    ang_df = pd.DataFrame({'Text': AngStandText, 'Origin': AngText})
    all_df = pd.concat([ang_df, eng_df, fr_df], ignore_index=True)

    all_df['Text'] = all_df['Text'].apply(lambda x: ' '.join(x))
    all_df['Origin'] = all_df['Origin'].apply(lambda x: ' '.join(x))

    all_ori_all_words = []
    for text in all_df['Origin']:
        tokens = text.split()
        all_ori_all_words.extend(tokens)

    all_stand_all_words = []
    for text in all_df['Text']:
        tokens = text.split()
        all_stand_all_words.extend(tokens)

    print("All ori", all_ori_all_words[:10][:10])
    print("All stand", all_stand_all_words[:10][:10])
    print("All ori", len(all_ori_all_words))
    print("All stand", len(all_stand_all_words))
    print("All ori distinct words", len(set(all_ori_all_words)))
    print("All stand distinct words", len(set(all_stand_all_words)))
    print("different words", len(set(all_ori_all_words)) - len(set(all_stand_all_words)))
    print("different words", len(set(all_ori_all_words)-set(all_stand_all_words)))




if __name__ == "__main__":
    standardization_p6()
    standardization_evaluation()



