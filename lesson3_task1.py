# TF-IDF

# TERM FREQUENCY - INVERSED DOCUMENT FREQUENCY d1, d2, d3 so on

# T is a set of all terms in the document

# t1 can be included in d1 n1 times
# t1 can be included in d2 n2 times

# len(d1) is how many words are present in d1
#
# term frequency for t1 in d1 is n1/len(d1)

# p1 is how many documents contain t at least once
# p is a total number of documents

# idf(t1) = log(p/p1)

# tfidf(t1) = tf(t1) * idf(t1)

from math import log
from sklearn.feature_extraction.text import CountVectorizer

def show(scores, words):
    s = sorted(zip(scores, [i for i in range(len(scores))]), reverse=True)
    for i in range(30):
        print(words[s[i][1]], s[i][0])



def read(name):
    res = ""
    with open(name,'r',encoding="utf-8") as f:
        for i in f.readlines():
            res = res + "" + i
    return res

documents = [
    read("document1.txt"),
    read("document2.txt"),
]



vect = CountVectorizer()
matrix = vect.fit_transform(documents)
city1 = matrix.toarray()[0]
city2 = matrix.toarray()[1]
words = vect.get_feature_names_out()
count = len(words)
city1_sum = sum(city1)
city1_res = [0.0 for i in range(count)]
city2_sum = sum(city2)
city2_res = [0.0 for i in range(count)]


for i in range(count):
    idf = 0.0
    if city1[i] > 0:
        idf = idf +1
    if city2[i] > 0:
        idf = idf +1
    idf = log(float(2) / idf)
    # print("+", city1[i], city1_sum)
    city1_res[i] = (float(city1[i]) / city1_sum) * idf
    city2_res[i] = (float(city2[i]) / city2_sum) * idf

show(city1, words)

import numpy as np


# 假设 city1_res 和 city2_res 是已经计算好的TF-IDF向量（列表或数组）
# 例如：
# city1_res = [0.1, 0.2, 0.0, ...]
# city2_res = [0.0, 0.2, 0.3, ...]

def cosine_similarity(vec1, vec2):
    # Convert a list to a NumPy array
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    # Calculating the dot product
    dot_product = np.dot(vec1, vec2)
    # Calculate the module length
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    # Avoid division by zero
    if norm1 == 0 or norm2 == 0:
        return 0.0
    else:
        return dot_product / (norm1 * norm2)

# Calculate the cosine similarity between two documents
similarity = cosine_similarity(city1_res, city2_res)
print(f"Cosine Similarity between documents: {similarity:.4f}")







