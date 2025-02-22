import numpy as np
import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample Data (Replace with actual dataset)
data = {
    "domain": ["example.com", "mali12cd.com", "abcdefg.com", "xzqweasd.net", "google.com"],
    "label": [0, 1, 0, 1, 0]  # 0 = Legitimate, 1 = DGA
}
df = pd.DataFrame(data)

# Function to preprocess domain names
def preprocess_domain(domain):
    return re.sub(r'[^a-zA-Z]', '', domain.lower())

df["clean_domain"] = df["domain"].apply(preprocess_domain)

# Feature Engineering - N-gram Tokenization
vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 6))
X = vectorizer.fit_transform(df["clean_domain"])
y = df["label"]

# Splitting Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training SVC Model
svc_model = SVC(kernel='linear', probability=True)
svc_model.fit(X_train, y_train)

# Evaluating SVC Model
y_pred_svc = svc_model.predict(X_test)
print("SVC Accuracy:", accuracy_score(y_test, y_pred_svc))
print(classification_report(y_test, y_pred_svc))

# Save the SVC model
joblib.dump(svc_model, 'svc_dga_detector.pkl')
joblib.dump(vectorizer, 'count_vectorizer.pkl')

# Neural Network Approach
max_words = 5000
max_length = 20
tokenizer = Tokenizer(num_words=max_words, char_level=True)
tokenizer.fit_on_texts(df["clean_domain"])

X_nn = tokenizer.texts_to_sequences(df["clean_domain"])
X_nn = pad_sequences(X_nn, maxlen=max_length)
X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(X_nn, y, test_size=0.2, random_state=42)

# Defining LSTM Model
model = Sequential([
    Embedding(max_words, 50, input_length=max_length),
    LSTM(64, return_sequences=True),
    LSTM(32),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train_nn, y_train_nn, epochs=5, batch_size=16, validation_data=(X_test_nn, y_test_nn))

# Save the LSTM model
model.save('lstm_dga_detector.h5')
