from flask import Flask, request, render_template
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from functools import lru_cache

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# Load dataset
df = pd.read_csv('data/dataset-wisata.csv')

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

# Main page endpoint
@app.route('/')
def home():
    return render_template('home.html')

# Recommendation endpoint
@app.route('/recommend', methods=['POST'])
def recommend():
    user_input = request.form['user_input']
    
    # Preprocess user input
    preprocessed_input = preprocess(user_input)

    # Drop rows with NaN values in 'city_category' column
    df_cleaned = df.dropna(subset=['city_category'])

    # Extract 'city_category' values as list
    city_categories = df_cleaned['city_category'].tolist()

    # Preprocess city categories
    preprocessed_city_categories = [preprocess(city_category) for city_category in city_categories]

    # Use TF-IDF for text representation on city_category
    tfidf_city = TfidfVectorizer(max_df=0.85, max_features=5000)
    tfidf_city_matrix = tfidf_city.fit_transform(preprocessed_city_categories)

    # Preprocess user input for city_category
    preprocessed_input_city = tfidf_city.transform([preprocessed_input])

    # Calculate cosine similarity on city_category
    cosine_sim_city = cosine_similarity(preprocessed_input_city, tfidf_city_matrix)

    # Sort tourist places based on city_category similarity
    sim_scores_city = list(enumerate(cosine_sim_city[0]))
    sim_scores_city = sorted(sim_scores_city, key=lambda x: x[1], reverse=True)

    # Get top recommended city_category
    top_sim_scores_city = sim_scores_city[:1]  # Only take the most similar city_category
    top_index_city = top_sim_scores_city[0][0]  # Get the index of the most similar city_category

    # Filter dataframe based on city_category
    similar_city_category = df_cleaned.iloc[top_index_city]['city_category']
    filtered_df = df_cleaned[df_cleaned['city_category'] == similar_city_category]
    
    # Handle NaN values in Description column
    filtered_df['Description'] = filtered_df['Description'].fillna('')

    # Combine filtered tourist places descriptions with user input
    descriptions = filtered_df['Description'].tolist() + [user_input]
    
    # Preprocess descriptions with caching
    preprocessed_descriptions = [preprocess(description) for description in descriptions]

    # Use TF-IDF for text representation on descriptions after filtering
    tfidf_description = TfidfVectorizer(max_df=0.85, max_features=5000)
    tfidf_description_matrix = tfidf_description.fit_transform(preprocessed_descriptions[:-1])

    # Preprocess user input for descriptions
    preprocessed_input_description = tfidf_description.transform([preprocessed_descriptions[-1]])

    # Calculate cosine similarity on descriptions
    cosine_sim_description = cosine_similarity(preprocessed_input_description, tfidf_description_matrix)

    # Sort tourist places based on description similarity
    sim_scores_description = list(enumerate(cosine_sim_description[0]))
    sim_scores_description = sorted(sim_scores_description, key=lambda x: x[1], reverse=True)

    # Get top recommended tourist places based on description similarity
    top_sim_scores_description = sim_scores_description[:5]
    top_indices_description = [i[0] for i in top_sim_scores_description]

    # Display recommendations
    recommended_places = filtered_df.iloc[top_indices_description]
    recommendations = recommended_places[['Place_Name', 'Description', 'Rating']].to_dict(orient='records')

    return render_template('recommend.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
