import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class SkipGramModel(nn.Module):
    """
    The Skip-Gram model for word embeddings.

    :param vocab_size: The size of the vocabulary.
    :param embed_dim: The dimensionality of the embeddings.
    :param device: The device (CPU or GPU) on which the model will run.

    Attributes:
        input_embed (torch.nn.Embedding): Embedding layer for both target and context words.
    """
    def __init__(self, vocab_size, embed_dim, device="cpu"):
        super(SkipGramModel, self).__init__()
        self.input_embed = nn.Embedding(vocab_size, embed_dim)
        self.device = device
        self.to(device)

    def forward(self, target, other):
        """
        Performs the forward pass of the SkipGramModel.

        Maps both target and context words to their embeddings and returns them.

        :param target: Tensor of indices for target words.
        :param other: Tensor of indices for context words.
        :return: A tuple of tensors containing the embeddings for target and context words, respectively.
        """
        target_embeds = self.input_embed(target)
        other_embeds = self.input_embed(other)
        return target_embeds, other_embeds


def train_model(model, data, optimizer, num_epochs=100, device="cpu"):
    """
    Trains the SkipGramModel using the specified dataset and optimizer.

    :param model: An instance of SkipGramModel to be trained.
    :param data: An iterable dataset of training samples, where each sample includes target, context, and label.
    :param optimizer: The optimizer to use for updating model weights.
    :param num_epochs: The number of epochs to train the model for.
    :param device: The device (CPU or GPU) on which to perform training.
    """
    for epoch in range(num_epochs):
        data_iterator = tqdm(data, desc=f"Epoch {epoch+1}/{num_epochs}")
        total_loss = 0
        for batch in data_iterator:
            target, other, label = batch
            target, other, label = target.to(device), other.to(device), label.to(device)
            optimizer.zero_grad()
            target_embed, other_embed = model(target, other)
            p = torch.sum(target_embed * other_embed, axis=1)
            pred = torch.sigmoid(p)
            loss = F.binary_cross_entropy_with_logits(pred, label.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            data_iterator.set_postfix({"Loss": total_loss / len(data)})