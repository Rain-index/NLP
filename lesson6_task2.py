import torch
from sklearn.svm._libsvm import predict
from torch.utils.data import Dataset
import os
import torch.nn as nn

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

        return input_tensor, output_tensor, word

    @property
    def country_names(self):
        return self.countries.copy()


class RNNExample(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):   # output_size:有多少个国家；input：有多少个字母，hidden：隐藏层的长度
        super(RNNExample, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size)
        self.cl = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        rnn_out, hidden = self.rnn(x)
        output = self.cl(hidden[0])
        output = self.softmax(output)
        return output




dataset = CountryDataset("NLP/names")
# print(dataset)

# View Country List
print(dataset.country_names)


# Get the first sample
input_tensor, output_tensor, word = dataset[0]
# input_vector = input_tensor.unsqueeze(1)
print(f'Name is {word}')
print(f"Input Shape: {input_tensor.shape}")
print(f"Output Tags: {output_tensor}")   # Country one-hot vector



input_size = input_tensor.size()[1]
print(f"Input Size: {input_size}")
hidden_size = 128
output_size = output_tensor.size()[0]
print(f"output Size: {output_size}")
model = RNNExample(input_size, hidden_size, output_size)

# result = model(input_tensor)
# print(result)


# 调整输入维度适配RNN模型（添加序列长度和batch维度）
# 原始维度： (sequence_length, num_letters)
# 需要格式： (sequence_length, batch_size, input_size)
input_vector = input_tensor.unsqueeze(1)  # 添加batch维度 -> (seq_len, 1, 52)

# 模型前向传播
with torch.no_grad():  # 禁用梯度计算
    model_output = model(input_vector)

# 结果解析
_, predicted_idx = torch.max(model_output, 1)
predicted_country = dataset.country_names[predicted_idx.item()]

# 打印处理结果
# print(f"Processing words: {word}")
print(f"Input Shape: {input_vector.shape} (sequence length x batch size x feature dimension)")
print(f"Forecast Country Index: {predicted_idx.item()}")
print(f"Predicted Country Name: {predicted_country}")


# 查看第一个样本的详细信息 //View details of the first sample
print("\nSample details verification:")
print(f"Original word: {word}")
print(f"Letter sequence length: {input_tensor.shape[0]}")
print(f"Corresponding country label: {torch.argmax(output_tensor).item()} - {dataset.country_names[torch.argmax(output_tensor).item()]}")


