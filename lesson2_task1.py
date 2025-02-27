import pandas as pd
import numpy as np
from nltk import word_tokenize, sent_tokenize

df = pd.read_csv('lesson2_data1.txt', sep='\t')
text = "On a sunny Saturday morning. I decided to visit the local park where children were laughing on the swings, dogs were chasing frisbees across the green grass, and elderly couples sat quietly on weathered wooden benches. Enjoying the gentle breeze that carried the scent of blooming lilacs from a nearby garden while I walked along the winding path, sipping my warm peppermint tea from a reusable thermos and occasionally stopping to watch ducks paddle peacefully in the small pond dotted with lily pads."
# print(type(text))
# print(type(df))
# print(df.head())

# word_tokenize函数用于分割单个词。
print(word_tokenize(text))
count = 0
for i in word_tokenize(text):
    if i == 'on':
        count += 1
print(count)

# sent_tokenize函数用于分割句子在“.”时分割

print(sent_tokenize(text))
for i in sent_tokenize(text):
    print(i)
    for j in word_tokenize(i):
        print(j)
    # print(i)


from nltk.stem import PorterStemmer, WordNetLemmatizer


text_2 = "On a sunny Saturday morning. I decided to visit the local park where children were laughing on the swings." #
# PoterStemmer 和 WordNetLemmatizer函数用于修正单词拼写错误
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()
# print(lemmatizer.lemmatize('on'))
text_3 = []
text_4 = []
# print(text_3)
for i in word_tokenize(text_2):
    text_3.append(ps.stem(i))
    text_4.append(lemmatizer.lemmatize(i))
    # print(ps.stem(i))
    # lemmatizer.lemmatize(ps.stem(i))
    # print(i)
print(f'PorterStemmer function result:{text_3}')
print(f'WordNetLemmatizer function result:{text_4}')
# print(lemmatizer.lemmatize(text_2))

# print(ps.stem("visites"))
