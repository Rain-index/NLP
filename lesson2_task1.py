
from nltk import word_tokenize, sent_tokenize

with open('lesson2_data1.txt', 'r', encoding='utf-8') as file:
    text = file.read()
    # for line in file:
    #     print(line.strip())  # 使用 strip() 去掉每行首尾的空白符

count = 0
for i in word_tokenize(text):
    if i == 'is':
        count += 1
print(count)



# text = """Our goal is to find what combination of variables can be used to make some sense out of this data, or to see if any of these variables have any meaningful impact. Since the data is about students, gpa may be a key variable that drives the relevance of the other 
# variables. The preceding image depicts scatter plots that show that a greater number 
# of female students have a higher gpa than the male students and a greater number of 
# male students spend more time on computer and get a similar gpa range of values. 
# Although all scatter plots are being shown here, the intent is to find out which data 
# plays a more significant role, and what sense can we make out of this data."""

# # print(text)
# # word_tokenize函数用于分割单个词。
# print(word_tokenize(text))
# count = 0
# for i in word_tokenize(text):
#     if i == 'is':
#         count += 1
# print(count)




