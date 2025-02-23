import pandas as pd
import gensim
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def preprocess_text(text):
    """Lowercase, remove stopwords, and tokenize."""
    tokens = text.lower().split()
    tokens = [word for word in tokens if word not in ENGLISH_STOP_WORDS]
    return tokens

if __name__ == "__main__":
    df = pd.read_csv("data/sentiment_data.csv")
    df["tokens"] = df["feedback"].apply(preprocess_text)

    # Train Word2Vec model
    word2vec_model = Word2Vec(sentences=df["tokens"], vector_size=100, window=5, min_count=1, workers=4)
    word2vec_model.save("src/features/word2vec.model")

    # Convert feedback into embeddings (average word vectors)
    def get_embedding(tokens):
        vectors = [word2vec_model.wv[word] for word in tokens if word in word2vec_model.wv]
        return sum(vectors) / len(vectors) if vectors else [0] * 100

    df["embedding"] = df["tokens"].apply(get_embedding)
    df_embeddings = df[["embedding", "sentiment"]].explode("embedding").pivot(index=df.index, columns="embedding")
    df_embeddings.to_csv("data/sentiment_embeddings.csv", index=False)

    print("Word2Vec embeddings created successfully.")
