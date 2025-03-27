import torch
from torch.utils.data import Dataset

class WordDataset(Dataset):
    def __init__(self, word_list):
        """
        Constructor :param word_list: input word list, repeats are allowed
        """
        self.words = word_list  # Original word list (duplicates allowed)
        self.vocab = sorted(list(set(word_list)))  # The vocabulary after deduplication
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}  # Mapping of words to indices
        self.one_hot_vectors = self._generate_one_hot_vectors()  # The generated one-hot encoded list

    def _generate_one_hot_vectors(self):
        """Generate one-hot encoded vectors for all words"""
        vectors = []
        vocab_size = len(self.vocab)
        for word in self.words:
            idx = self.word_to_idx[word]
            one_hot = torch.zeros(vocab_size, dtype=torch.float32)
            one_hot[idx] = 1.0
            vectors.append(one_hot)
        return vectors

    def __len__(self):
        """Returns the total size of the dataset (the number of raw words)"""
        return len(self.words)

    def __getitem__(self, idx):
        """Returns the one-hot encoding of a tensor"""
        return self.one_hot_vectors[idx]

    @property
    def vocab_size(self):
        """Returns the vocabulary size (number of unique words)"""
        return len(self.vocab)




# Create a dataset
word_list = ["apple", "banana", "apple", "orange"]
dataset = WordDataset(word_list)

# View vocabulary size
print(f'Vocabulary size:{dataset.vocab_size}')  # 3

# Accessing data
print(dataset[0])  # tensor([1., 0., 0.]) corresponds to "apple"
print(dataset[1])  # tensor([0., 1., 0.]) corresponds to "banana"
print(dataset[2])  # tensor([1., 0., 0.]) corresponds to "apple"
print(dataset[3])  # tensor([0., 0., 1.]) corresponds to "orange"

# Get all one-hot encodings
all_vectors = [dataset[i] for i in range(len(dataset))]
print(all_vectors)