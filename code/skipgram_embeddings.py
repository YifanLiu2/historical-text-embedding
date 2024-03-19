import os
from copy import deepcopy
from gensim.models import Word2Vec, FastText
from tqdm import tqdm
import numpy as np


def read_corpus(file_path):
    with open(file_path) as f:
        for line in f:
            yield line.strip().split()


#### model2: fasttext ####
def train_fasttext(file_path="inputs/AllStandText", output_dir="outputs/", model_path="fasttext.model"):
    """
    Train the  `fastText` subword model from scratch.
    """
    params = {
        'vector_size': 100, 
       # 'window': 5,
       # 'min_count': 5,
        'workers': 8, 
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

    corpus = list(tqdm(read_corpus(file_path), desc="Reading Corpus"))
    model = FastText(corpus, **params)

    print("model trained")
    merged_model_path = os.path.join(output_dir, model_path)
    
    model.save(merged_model_path)


#### model3: fasttext + continue training ####
def continue_train(pre_trained_path="outputs/fasttext.model", file_path="inputs/AngOrdtext", output_dir="outputs/",
                   model_path="adapt.model"):
    """
    Continue training the pretrained `fastText` modern Latin embedding.
    """
    print("continue_train")
    model = FastText.load(pre_trained_path)
    
    corpus = list(tqdm(read_corpus(file_path), desc="Reading Corpus"))

    model.build_vocab(corpus, update=True)

    model.train(corpus, total_examples=len(corpus), epochs=30)

    merged_model_path = os.path.join(output_dir, model_path)
    
    model.save(merged_model_path)

def evaluation():
    model_0 = FastText.load("outputs/fasttext.model")
    model_1 = FastText.load("outputs/ang_adapt.model")
    model_2 = FastText.load("outputs/eng_adapt.model")
    print("Model 0 vocab size:", len(model_0.wv.index_to_key))
    print("Model 1 vocab size:", len(model_1.wv.index_to_key))
    print("Model 2 vocab size:", len(model_2.wv.index_to_key))
    word = "presenstes"
    if word in model_0.wv.index_to_key:
        print(word + " in model 0")
    if word in model_1.wv.index_to_key:
        print(word + " in model 1")
    if word in model_2.wv.index_to_key:
        print(word + " in model 2")
    if word in model_0.wv.index_to_key and word in model_1.wv.index_to_key and word in model_2.wv.index_to_key:
        print(np.allclose(model_0.wv[word], model_1.wv[word], atol=1e-5))
        print(np.allclose(model_0.wv[word], model_2.wv[word],atol=1e-5))
        print(np.allclose(model_1.wv[word], model_2.wv[word],atol=1e-5))

    print(model_0.wv.key_to_index["presenstes"])
    print(model_1.wv.key_to_index["presenstes"])
    print(model_2.wv.key_to_index["presenstes"])


def run():
    train_fasttext(file_path="inputs/AngStandText", output_dir="outputs/", model_path="ang_fasttext.model")
    # continue_train(file_path="inputs/AngStandText", output_dir="outputs/", model_path="ang_adapt.model")
    continue_train(file_path="inputs/EngStandText", output_dir="outputs/", model_path="eng_from_ang.model")

def test():
    test = [
        ["wow", "this", "is", "a", "test"],
        ["this", "is", "another", "test"],
        ["this", "is", "a", "test"]
    ]
    model = FastText(vector_size=10, 
                     workers=8, 
                     sg=0, hs=0, 
                     negative=5, 
                     ns_exponent=0.75, 
                     alpha=0.025, 
                     min_alpha=0.0001, 
                     epochs=30, 
                     sample=1e-3, 
                     min_n=5, max_n=5, 
                     bucket=2000000)
    
    model.build_vocab(test)
    model.train(test, total_examples=len(test), epochs=30)
    
    
    print(model.wv["this"])

def evaluation_2():
    model_0 = FastText.load("outputs/ang_fasttext.model")
    model_1 = FastText.load("outputs/eng_from_ang.model")
    print("Model 0 vocab size:", len(model_0.wv.index_to_key))
    print("Model 1 vocab size:", len(model_1.wv.index_to_key))
    word = "presenstes"
    if word in model_0.wv.index_to_key:
        print(word + " in model 0")
    if word in model_1.wv.index_to_key:
        print(word + " in model 1")
    if word in model_0.wv.index_to_key and word in model_1.wv.index_to_key:
        print(np.allclose(model_0.wv[word], model_1.wv[word], atol=1e-5))

    print(model_0.wv.key_to_index["presenstes"])
    print(model_1.wv.key_to_index["presenstes"])

if __name__ == "__main__":
    evaluation_2()