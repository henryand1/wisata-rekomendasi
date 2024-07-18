from flask import Flask, request, jsonify
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from functools import lru_cache

# Download resource nltk
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# Load dataset
df = pd.read_csv('data/dataset-wisata-new.csv')

# Initialize Stemmer for Bahasa Indonesia
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Cache Indonesian stop words
stop_words = set(stopwords.words('indonesian'))

# Preprocessing function with caching to avoid repetitive processing
@lru_cache(maxsize=None)
def preprocess(text):
    # Tokenization and Lowercasing
    tokens = [word.lower() for word in word_tokenize(text)]
    
    # Removing punctuation and stop words removal
    tokens = [re.sub(r'\W+', '', word) for word in tokens if word not in stop_words and re.sub(r'\W+', '', word)]
    
    # Stemming
    tokens = [stemmer.stem(word) for word in tokens]
    
    return ' '.join(tokens)

# Preprocess and fit TF-IDF for city categories
df = df.dropna(subset=['city_category'])
df['preprocessed_city_category'] = df['city_category'].apply(preprocess)
tfidf_city = TfidfVectorizer(max_df=0.85, max_features=5000)
tfidf_city_matrix = tfidf_city.fit_transform(df['preprocessed_city_category'])

# Recommendation endpoint
@app.route('/recommend', methods=['POST'])
def recommend():
    user_input = request.json.get('user_input')
    if not user_input:
        return jsonify({"error": "user_input is required"}), 400
    
    # Preprocess user input
    preprocessed_input = preprocess(user_input)

    # Transform user input for city_category
    preprocessed_input_city = tfidf_city.transform([preprocessed_input])

    # Calculate cosine similarity on city_category
    cosine_sim_city = cosine_similarity(preprocessed_input_city, tfidf_city_matrix)

    # Sort tourist places based on city_category similarity
    sim_scores_city = list(enumerate(cosine_sim_city[0]))
    sim_scores_city = sorted(sim_scores_city, key=lambda x: x[1], reverse=True)

    # Get top recommended city_category
    top_index_city = sim_scores_city[0][0]  # Most similar city_category

    # Filter dataframe based on city_category
    similar_city_category = df.iloc[top_index_city]['city_category']
    filtered_df = df[df['city_category'] == similar_city_category]
    
    # Preprocess and fit TF-IDF for descriptions of filtered data
    filtered_df['preprocessed_description'] = filtered_df['Description'].apply(preprocess)
    tfidf_description = TfidfVectorizer(max_df=0.85, max_features=5000)
    tfidf_description_matrix = tfidf_description.fit_transform(filtered_df['preprocessed_description'])

    # Transform user input for descriptions
    preprocessed_input_description = tfidf_description.transform([preprocessed_input])

    # Calculate cosine similarity on descriptions
    cosine_sim_description = cosine_similarity(preprocessed_input_description, tfidf_description_matrix)

    # Sort tourist places based on description similarity
    sim_scores_description = list(enumerate(cosine_sim_description[0]))
    sim_scores_description = sorted(sim_scores_description, key=lambda x: x[1], reverse=True)

    # Get top recommended tourist places based on description similarity
    top_indices_description = [i[0] for i in sim_scores_description[:5]]

    # Display recommendations
    recommended_places = filtered_df.iloc[top_indices_description]
    recommendations = recommended_places[['Place_Name', 'Description', 'Rating', 'City']].to_dict(orient='records')

    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)
