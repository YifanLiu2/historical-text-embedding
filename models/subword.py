import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class SubwordModel(nn.Module):
    """
    The Skip-gram model with subword information. 

    :param vocab_size: The size of the word vocabulary.
    :param ngram_vocab_size: The size of the subword vocabulary.
    :param embed_dim: The dimensionality of the embeddings.
    :param device: The device (CPU or GPU) on which the model will run.

    Attributes:
        word_embed (torch.nn.Embedding): Embedding layer for word-level embeddings.
        subword_embed (torch.nn.Embedding): Embedding layer for subword-level embeddings.
    """
    def __init__(self, vocab_size, ngram_vocab_size, embed_dim, device='cpu'):
        self.word_embed = nn.Embedding(vocab_size, embed_dim)
        self.subword_embed = nn.Embedding(ngram_vocab_size, embed_dim)
        self.device = device
        self.to(device)
    
    def forward(self, target, other, target_sub, other_sub):
        """
        Forward pass of the SubwordModel.

        Combines word-level and subword-level embeddings by summing them to generate the final embeddings.

        :param target: Tensor of target word indices.
        :param other: Tensor of context word indices.
        :param target_sub: Tensor of subword indices for the target words.
        :param other_sub: Tensor of subword indices for the context words.
        :return: A tuple containing the combined embeddings for the target and context words.
        """
        target_embed = self.word_embed(target)
        other_embed = self.word_embed(other)

        target_sub_embed = self.subword_embed(target_sub).sum(dim=1)
        other_sub_embed = self.subword_embed(other_sub).sum(dim=1)

        target_embed = target_embed + target_sub_embed
        other_embed = other_embed + other_sub_embed

        return target_embed, other_embed

def train_model(model, data, optimizer, num_epochs=100, device="cpu"):
    """
    Trains the SubwordModel using the given data and optimizer.

    :param model: An instance of SubwordModel to be trained.
    :param data: An iterable dataset of training samples.
    :param optimizer: The optimizer to use for updating model weights.
    :param num_epochs: The number of epochs to train the model for.
    :param device: The device (CPU or GPU) on which to perform training.
    """
    for epoch in range(num_epochs):
        data_iterator = tqdm(data, desc=f"Epoch {epoch+1}/{num_epochs}")
        total_loss = 0
        for batch in data_iterator:
            target, other, target_sub, other_sub, label = batch
            target, other, target_sub, other_sub, label = target.to(device), other.to(device), target_sub.to(device), other_sub.to(device), label.to(device)
            target_embed, other_embed = model(target, other, target_sub, other_sub)
            p = torch.sum(target_embed * other_embed, axis=1)
            pred = torch.sigmoid(p)
            loss = F.binary_cross_entropy_with_logits(pred, label.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            data_iterator.set_postfix({"Loss": total_loss / len(data)})