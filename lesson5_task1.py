

from nltk import word_tokenize, sent_tokenize
import torch

text = "Traditional arts thrive alongside modernity. In the ancient village of Ciqikou, artisans carve porcelain and weave silk, preserving techniques from the Ming and Qing dynasties. The city’s iconic *Diaojiaolou*—wooden stilted houses clinging to cliffs—offer glimpses of pre-industrial life, though many are now repurposed as teahouses or boutique hotels. Chongqing’s dialect, a melodic variant of Mandarin, further distinguishes its local identity, often described as loud but warm by outsiders.  "

def tokenize_onehot(text):
    # Split words
    tokens = []

    for i in sent_tokenize(text):
        for j in word_tokenize(i):
            tokens.append(j)
    # print(tokens)

    # Creating a vocabulary dictionary (words_dict)
    words_dict = {}
    index = 0
    for token in tokens:
        if token not in words_dict:
            words_dict[token] = index
            index += 1

    # Generate one-hot encoding
    one_hot_vectors = torch.zeros(len(words_dict), len(words_dict))
    for key in words_dict:
        one_hot_vectors[words_dict[key]][words_dict[key]] = 1



    return tokens, words_dict, one_hot_vectors

tokens, words_dict, one_hot_vectors = tokenize_onehot(text)

print("Tokens word: ", tokens)

# print(words_dict)
print("\nThe words dict: ")
for word, index in words_dict.items():
    print(word, ":", index)

# print(one_hot_vectors)

print("\nThe one_hot_vectors:")
for token, one_hot in zip(tokens, one_hot_vectors):
    print(token, ":", one_hot)



    # print(len(words_dict))
    # print(words_dict)
# print(tokenize(text))







