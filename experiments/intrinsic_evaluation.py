from gensim.models import KeyedVectors
from gensim.test.utils import datapath


# Load word static embeddings learned from Norman period
model_path = f"eng_model.bin" 
model_norman = KeyedVectors.load_word2vec_format(model_path, binary=False) 

input_path = f"/Users/carolinechen/Documents/GitHub/historical-text-embedding/experiments/text_vocab.txt"
output_path = f"similar_words.txt"

# Load 20 representative words
with open(input_path) as input_file:
    words = [line.strip() for line in input_file]

with open(output_path, 'w') as outfile:
    # Calculate by cosine similarity to get the top ten most similar words to each representative word 
    for word in words:
        if word in model_norman:
            similar = model_norman.most_similar(word, topn=10)
            similar_words_str = ', '.join([f"{sim_word} ({similarity:.2f})" for sim_word, similarity in similar]) # similar word and its similarity score
            outfile.write(f'Top ten similar words to "{word}" from Norman period are: {similar_words_str}\n')
        else:
            outfile.write(f'"{word}" not in corpus.\n')
