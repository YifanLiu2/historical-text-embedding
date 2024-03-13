from pretrain_tokenizer import *
from bert_embeddings import *
from extract_embeddings import *


def main(args):
    if not os.path.exists(args.out):
        os.makedirs(args.out)

    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input corpus file not found at {args.input}")

    # pretrain tokenizer
    train_tokenizer(args.input, args.out)
    
    # further train BERT model
    train_BERT(args.model, args.input, args.out, args.pretrain)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adapt BERT for medieval Latin corpora and extract embeddings.")
    parser.add_argument("--input", type=str, default="inputs/corpus.txt", help="Path to the corpus file.")
    parser.add_argument("--out", type=str, default="outputs/", help="Directory to save the adapted model, tokenizer, and embeddings.")
    parser.add_argument("--model", type=str, default="bert-base-multilingual-cased", help="BERT model to use.")
    parser.add_argument("--pretrain", type=bool, default=True, help="Whether to use pretraining mode or fine-tune mode.")
    parser.add_argument("--train_tokenizer", type=bool, default=True, help="Whether to pretrain the tokenizer.")
    
    args = parser.parse_args()
   
    main(args)