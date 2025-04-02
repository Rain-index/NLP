import torch
from torch.utils.data import Dataset
import os


class CountryDataset(Dataset):
    def __init__(self, directory):

        self.country_files = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.endswith('.txt') and os.path.isfile(os.path.join(directory, f))
        ]

        # Constructing the country list
        self.countries = [
            os.path.splitext(os.path.basename(f))[0]
            for f in self.country_files
        ]
        self.countries.sort()
        self.country_to_idx = {country: idx for idx, country in enumerate(self.countries)}
        self.num_countries = len(self.countries)

        self.letters = (
                [chr(ord('a') + i) for i in range(26)] +  # 小写字母
                [chr(ord('A') + i) for i in range(26)]  # 大写字母
        )
        self.letter_to_idx = {char: idx for idx, char in enumerate(self.letters)}
        self.num_letters = len(self.letters)  # 固定52个字符

        # Read all words and record the labels
        self.words = []
        self.labels = []
        for country_file in self.country_files:
            name = os.path.splitext(os.path.basename(country_file))[0]
            with open(country_file, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip()
                    if word:  # 跳过空行
                        self.words.append(word)
                        self.labels.append(self.country_to_idx[name])

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        word = self.words[idx]
        country_idx = self.labels[idx]

        # Generate letter sequence
        sequence = []
        for char in word:
            if char in self.letter_to_idx:
                one_hot = torch.zeros(self.num_letters, dtype=torch.float32)
                one_hot[self.letter_to_idx[char]] = 1.0
                sequence.append(one_hot)

        # Handling empty words
        if not sequence:
            sequence.append(torch.zeros(self.num_letters, dtype=torch.float32))


        input_tensor = torch.stack(sequence)


        output_tensor = torch.zeros(self.num_countries, dtype=torch.float32)
        output_tensor[country_idx] = 1.0

        return input_tensor, output_tensor,word

    @property
    def country_names(self):
        return self.countries.copy()


dataset = CountryDataset("name")
# print(dataset)

# View Country List
print(dataset.country_names)

dataset = CountryDataset("name")

# Get the first sample
input_tensor, output_tensor, word = dataset[0]
print(f'Name is {word}')
print(f"Input Shape: {input_tensor.shape}")
print(f"Output Tags: {output_tensor}")   # Country one-hot vector






