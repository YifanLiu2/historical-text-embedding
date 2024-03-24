# Diachronic Embeddings for Medieval Latin

This project is dedicated to adapting various models, such as FastText and BERT, for Medieval Latin corpora, and evaluating their performance through both extrinsic and intrinsic methods.

## Setup

## Usage

The project includes the training and adaptation of models, evaluation of model performance, and a example semantic change detection analysis.

### Adapt Models and Extract Embeddings

#### FastText Embeddings

For FastText model adaptation and embedding extraction:


#### Adapt BERT Models

To adapt BERT for your medieval Latin corpus, use the `runner.py` script in `code/bert`:

- **Adapt BERT with an existing tokenizer and model:**
    ```bash
    python code/bert/runner.py -i path/to/corpus.txt -o path/to/output
    ```

- **Adapt BERT with tokenizer training:**
    ```bash
    python code/bert/runner.py -i path/to/corpus.txt -o path/to/output --train_tokenizer
    ```

- **Pretrain BERT from scratch:**
    ```bash
    python code/bert/runner.py -i path/to/corpus.txt -o path/to/output --pretrain
    ```
- **Extract word embeddings from BERT models:**
  ```bash
  python code/bert/extract_embeddings.py -t /path/to/pretrained/tokenizer -m /path/to/pretrained/models -c /path/to/corpus/files -o /path/to/save/embeddings
  ```

### Evaluate Models

#### Extrinsic Evaluations through Text Classification

To extrinsically evaluate models on text classification tasks, use the `text_classification.py` script located in `code/evaluation`. Ensure the corpus and corresponding labels are aligned and of the same length.


--**Evaluate all files from a metadata directory:**--
  ```bash
  python code/evaluation/text_classification.py -m path/to/model -c path/to/corpus.txt -ld path/to/label_dir -o path/to/output
  ```

--**Evaluate a single metadata file:**--
  ```bash
  python code/evaluation/text_classification.py -m path/to/model -c path/to/corpus.txt -ld path/to/label_dir -o path/to/output -l specific_label_file_name.txt
  ```

