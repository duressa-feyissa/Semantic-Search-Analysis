# Semantic Analysis Query System

This is a Flask-based web application for performing semantic analysis and querying a pre-processed dataset. The application uses NLP techniques to process text and retrieve similar documents based on cosine similarity.

## Features

- **Text Preprocessing**: Tokenization, stopword removal, stemming, and lemmatization.
- **Semantic Querying**: Find similar documents using TF-IDF vectors and cosine similarity.
- **Named Entity Recognition**: Identify entities in text using spaCy.

## Requirements

To install the required Python packages, use the `requirements.txt` file:

```plaintext
flask
pandas
nltk
spacy
scikit-learn
joblib
```

## DataSet
- [Download Dataset](https://drive.google.com/file/d/1lzZSXcENB1MpQsHrA2nRGXiU4m0m-dEu/view?usp=sharing)

Install the dependencies with the following command:

```
pip install -r requirements.txt
```

## Live Demo
- [Link](https://semantic-search-analysis.onrender.com)



## Usage
### Running the Application
Run the Flask application using the following command:

```python app.py```


## Accessing the Web Interface
Open your web browser and go to http://127.0.0.1:5000 to access the query interface.

### Querying the System
- Enter your query in the input field.
- Click the "Submit" button.
- View the results displayed below the form.

## Model File
- semantic_search_analysis.py
- Semantic_Search_Analysis.ipynb
