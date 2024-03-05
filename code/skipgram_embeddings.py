import os
from copy import deepcopy
from gensim.models import Word2Vec, FastText


def read_corpus(file_path):
    with open(file_path) as f:
        for line in f:
            yield line.strip().split()


#### model1: word2vec ####
def train_word2vec(ang_file_path="inputs/AngOrdtext", eng_file_path="inputs/EngOrdtext", output_dir="outputs/"):
    """
    Train the `word2vec` skipgram model from scratch.
    """
    params = {
        'vector_size': 100, 
        'window': 5, 
        'min_count': 5, 
        'workers': 4, 
        'sg': 0, 
        'hs': 0, 
        'negative': 5, 
        'ns_exponent': 0.75,
        'alpha': 0.025, 
        'min_alpha': 0.0001,
        'epochs': 30,
        'sample': 1e-3, 
    }

    ang_corpus = list(read_corpus(ang_file_path))
    eng_corpus = list(read_corpus(eng_file_path))
    
    ang_model = Word2Vec(ang_corpus, **params)
    eng_model = Word2Vec(eng_corpus, **params)

    ang_model_path = os.path.join(output_dir, "Ang_wordwvec.model")
    eng_model_path = os.path.join(output_dir, "Eng_wordwvec.model")
    
    ang_model.save(ang_model_path)
    eng_model.save(eng_model_path)


#### model2: fasttext ####
def train_fasttext(ang_file_path="inputs/AngOrdtext", eng_file_path="inputs/EngOrdtext", output_dir="outputs/"):
    """
    Train the  `fastText` subword model from scratch.
    """
    params = {
        'vector_size': 100, 
        'window': 5,
        'min_count': 5,
        'workers': 4, 
        'sg': 0, 
        'hs': 0, 
        'negative': 5, 
        'ns_exponent': 0.75, 
        'alpha': 0.025,  
        'min_alpha': 0.0001,  
        'epochs': 30,
        'sample': 1e-3, 
        'min_n': 5, 
        'max_n': 5, 
        'bucket': 2000000, 
    }

    ang_corpus = list(read_corpus(ang_file_path))
    eng_corpus = list(read_corpus(eng_file_path))
    
    ang_model = Word2Vec(ang_corpus, **params)
    eng_model = Word2Vec(eng_corpus, **params)

    ang_model_path = os.path.join(output_dir, "Ang_fasttext.model")
    eng_model_path = os.path.join(output_dir, "Eng_fasttext.model")
    
    ang_model.save(ang_model_path)
    eng_model.save(eng_model_path)


#### model3: fasttext + continue training ####
def continue_train(pre_trained_path="inputs/cc.la.100.model", ang_file_path="inputs/AngOrdtext", eng_file_path="inputs/EngOrdtext", output_dir="outputs/"):
    """
    Continue training the pretrained `fastText` modern Latin embedding.
    """
    ang_model = FastText.load(pre_trained_path)
    eng_model = deepcopy(ang_model)
    
    ang_corpus = list(read_corpus(ang_file_path))
    eng_corpus = list(read_corpus(eng_file_path))

    ang_model.build_vocab(ang_corpus, update=True)
    eng_model.build_vocab(eng_model, update=True)

    ang_model.train(ang_corpus, total_examples=len(ang_corpus), epochs=30)
    ang_model.train(eng_corpus, total_examples=len(eng_corpus), epochs=30)

    ang_model_path = os.path.join(output_dir, "Ang_adapt.model")
    eng_model_path = os.path.join(output_dir, "Eng_adapt.model")
    
    ang_model.save(ang_model_path)
    eng_model.save(eng_model_path)