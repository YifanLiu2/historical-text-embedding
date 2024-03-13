def read_corpus(file_path):
    """
    Generator that yields sentences from a corpus file, splitting by lines and then by spaces.
    
    :param file_path: Path to the corpus file.
    :return: Yields lists of words in each line.
    """
    with open(file_path) as f:
        for line in f:
            yield line.strip().split()


def chunk_text(text, tokenizer, max_length=512):
    """
    Splits a text into chunks that fit within the specified maximum length for BERT inputs.
    
    :param text: Text (list of words) to chunk.
    :param tokenizer: Tokenizer to use for determining word token lengths.
    :param max_length: Maximum length of tokens in a chunk.
    :return: List of text chunks, each represented as a single string.
    """
    words = text
    word_token_len = [len(tokenizer.tokenize(word)) for word in words]

    chunks = []
    chunk = []
    curr_len = 0
    for word, len_ in zip(words, word_token_len):
        if curr_len + len_ + 2 > max_length:
            chunks.append(" ".join(chunk))
            chunk = []
            curr_len = 0
        chunk.append(word)
        curr_len += len_

    if chunk:
        chunks.append(" ".join(chunk))

    return chunks