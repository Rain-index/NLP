
import gensim.downloader as api

model = api.load("glove-twitter-200")
# model = api.load("word2vec-google-news-300")
# model = api.load("fasttext-wiki-news-subwords-300")

positive = ["bright", "road", "drive", "summer", "sunshine", "sunny", "summer", "beach", "roadtrip", "driving", "highway", "convertible",]   # 亮度、白天、太阳的
negative = ["night", "walk", "rain"]           # 黑暗、夜晚

sims = model.most_similar(positive=positive, negative=negative, topn=10)

for i in range(10):
    print(sims[i][0])

