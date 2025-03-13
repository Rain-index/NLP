
import gensim.downloader as api

model = api.load("glove-twitter-200")
# model = api.load("word2vec-google-news-300")
# model = api.load("fasttext-wiki-news-subwords-300")

positive = ['brightness', 'daytime', 'solar', 'day', 'sunshine', 'sunlight']   # 亮度、白天、太阳的
negative = ['darkness', 'night', 'moon', 'dusk', 'twilight']                # 黑暗、夜晚

sims = model.most_similar(positive=positive, negative=negative, topn=10)

for i in range(10):
    print(sims[i][0])

