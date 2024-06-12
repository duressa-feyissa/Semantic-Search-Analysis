from flask import Flask, request, jsonify, render_template
import pandas as pd
import nltk
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from spacy.cli import download

app = Flask(__name__)

nltk.download('punkt')
nltk.download('names')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

download("en_core_web_sm")

nlp = spacy.load('en_core_web_sm')

# Load saved model and data
vectorizer = joblib.load('vectorizer.joblib')
df = pd.read_pickle('processed_data.pkl')

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\b\w{1,2}\b', '', text)
    text = re.sub(r'\d+', '', text)
    words = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

def query_system(query, vectorizer, tfidf_matrix, df, top_k=5):
    preprocessed_query = preprocess_text(query)
    query_vector = vectorizer.transform([preprocessed_query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    results = [(df.iloc[i]['id'], df.iloc[i]['text'], similarities[i]) for i in top_indices]
    return results

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    query = data.get('query', '')
    if not query:
        return jsonify({'error': 'No query provided'}), 400

    results = query_system(query, vectorizer, vectorizer.transform(df['processed_text']), df)
    response = [result[1] for result in results]
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
