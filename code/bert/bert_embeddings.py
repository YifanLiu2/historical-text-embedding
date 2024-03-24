import os
from transformers import BertConfig, BertForMaskedLM, BertTokenizerFast, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import Dataset

from ..utils import *


def train(model, output_path, data_collator, dataset, epochs):
    """
    Trains the BERT model using the provided DataLoader, learning rate, number of epochs, and device.
    :param mode;: The BERT model to train.
    :param model_path: Path to save the BERT model
    :param data_collator: An instance of a data collator that prepares batches of data during training.
    :param dataset: The dataset to be used for training, an insance of datasets.Dataset.
    :param epochs: Number of training epochs.
    """
    # create output dir
    os.makedirs(output_path, exist_ok=True)

    # define training arguments
    training_args = TrainingArguments(
        overwrite_output_dir = True,
        output_dir=output_path,
        num_train_epochs=epochs, # 4 for fine-tune; 10 for pretrain
        per_device_train_batch_size=8,
        gradient_accumulation_steps=8,
        per_device_eval_batch_size=8,
        logging_steps=1000,
        save_steps=10000,
        save_total_limit=2, 
    )  
    
    # define trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # train and save model
    trainer.train()
    trainer.save_model(output_path)


def train_BERT(model_name, input_file, output_dir, is_pretraining):
    """
    Adapts BERT for medieval Latin corpora by further pretraining on provided text files, with customizable parameters.

    :param params: Dictionary containing training parameters such as batch size, learning rate, epochs, and device.
    :param model_name: Name of the BERT model to use.
    :param input_file: Path to the file containing the corpus.
    :param output_dir: Directory to save the adapted model and tokenizer.
    :param is_pretraining: A boolean flag indicating whether the task is pretraining. If `True`, the model will be trained from scratch If `False`, the model will be fine-tuned.
    """
    # load pretrained tokenizer
    tokenizer_path = os.path.join(output_dir, "pretrained-tokenizer")
    try:
        tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer: {e}")
    
    # process corpora
    corpus = list(read_corpus(input_file))
    chunks = [chunk for doc in corpus for chunk in chunk_text(doc, tokenizer)]
    
    tokenized = tokenizer(
        chunks,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    dataset = Dataset.from_dict(tokenized)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.2)

    if is_pretraining:
        # pre-train BERT from scratch
        config = BertConfig(
            vocab_size=32_000,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            max_position_embeddings=512,
        )
        print("pre-train BERT model...")
        model = BertForMaskedLM(config=config)
        model_path = os.path.join(output_dir, "pretrained-bert")
        train(model, output_path=model_path, data_collator=data_collator, dataset=dataset, epochs=10)

    else:
        # fine-tune BERT model
        try:
            model = BertForMaskedLM.from_pretrained(model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
        print(f"fine-tuned BERT model: {model_name}...")
        
        # adjust tokenizer size for model fine-tune
        model.resize_token_embeddings(len(tokenizer))

        # define the output path to save model
        format_model_name = model_name.replace("/", "-")
        model_path = os.path.join(output_dir, f"fine-tuned-bert-{format_model_name}")
        train(model, output_path=model_path, data_collator=data_collator, dataset=dataset, epochs=4)

