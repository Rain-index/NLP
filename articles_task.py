import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# 1. 读取文章
def load_articles(folder_path):
    articles = {}
    filenames = []
    contents = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                content = f.read()
                articles[filename] = content
                filenames.append(filename)
                contents.append(content)

    return articles, filenames, contents


# 2. 初始化TF-IDF向量器
def init_tfidf(contents):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
    tfidf_matrix = vectorizer.fit_transform(contents)
    return vectorizer, tfidf_matrix


# 3. 查找最相关文章
def find_most_relevant_article(query, vectorizer, tfidf_matrix, filenames, contents):
    # 转换查询为TF-IDF向量
    query_vec = vectorizer.transform([query])

    # 计算余弦相似度
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # 获取最相关文章的索引
    most_relevant_idx = np.argmax(similarities)
    return filenames[most_relevant_idx], contents[most_relevant_idx], similarities[most_relevant_idx]


# 4. 加载Hugging Face模型和tokenizer
def load_model():
    model_name = "google/flan-t5-large"  # 选择Flan-T5模型
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model


# 5. 生成回答
def generate_answer(tokenizer, model, context, query, max_length=512):
    # 构建输入文本
    input_text = f"Read this article and answer the question:\n\nArticle: {context}\n\nQuestion: {query}\nAnswer:"

    # 编码输入
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    )

    # 生成输出
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=max_length,
        temperature=0.7,
        num_beams=3
    )

    # 解码输出
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer


# 主流程
def main():
    # 配置参数
    ARTICLES_FOLDER = "articles"
    TEST_QUERIES = [
        "Explain quantum computing basics",
        "Impact of Renaissance art",
        "Latest blockchain applications"
    ]

    # 加载文章
    articles, filenames, contents = load_articles(ARTICLES_FOLDER)

    # 初始化TF-IDF
    vectorizer, tfidf_matrix = init_tfidf(contents)

    # 加载模型
    tokenizer, model = load_model()

    # 测试查询
    for query in TEST_QUERIES:
        print(f"\n{'=' * 50}\nQuery: {query}\n{'=' * 50}")

        # 查找最相关文章
        filename, content, score = find_most_relevant_article(
            query, vectorizer, tfidf_matrix, filenames, contents
        )
        print(f"Selected article: {filename} (Score: {score:.4f})")

        # 显示文章片段
        print(f"\nArticle snippet: {content[:200]}...")

        # 生成回答
        answer = generate_answer(tokenizer, model, content, query)
        print(f"\nGenerated Answer: {answer}")


if __name__ == "__main__":
    main()