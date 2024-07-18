from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

app = Flask(__name__)

# Load dataset
df = pd.read_csv('data/dataset-wisata-new.csv')

# Basic text preprocessing function
def preprocess(text):
    # Lowercasing
    text = text.lower()
    # Removing punctuation and non-alphabetic characters
    text = re.sub(r'\W+', ' ', text)
    # Removing extra spaces
    text = text.strip()
    return text

# Preprocess city categories and descriptions
df['cleaned_city_category'] = df['city_category'].apply(preprocess)
df['cleaned_description'] = df['Description'].apply(preprocess)

# Fit TF-IDF vectorizer for city categories
tfidf_city = TfidfVectorizer()
tfidf_city_matrix = tfidf_city.fit_transform(df['cleaned_city_category'])

# Fit TF-IDF vectorizer for descriptions
tfidf_description = TfidfVectorizer()
tfidf_description_matrix = tfidf_description.fit_transform(df['cleaned_description'])

# Recommendation endpoint
@app.route('/recommend', methods=['POST'])
def recommend():
    user_input = request.json.get('user_input')
    if not user_input:
        return jsonify({"error": "user_input is required"}), 400
    
    # Preprocess user input
    preprocessed_input = preprocess(user_input)

    # Transform user input for city_category
    input_city_tfidf = tfidf_city.transform([preprocessed_input])

    # Calculate cosine similarity on city_category
    cosine_sim_city = cosine_similarity(input_city_tfidf, tfidf_city_matrix)
    top_index_city = cosine_sim_city.argmax()

    # Filter dataframe based on most similar city_category
    similar_city_category = df.iloc[top_index_city]['city_category']
    filtered_df = df[df['city_category'] == similar_city_category]

    # Transform user input for descriptions
    input_description_tfidf = tfidf_description.transform([preprocessed_input])

    # Calculate cosine similarity on descriptions
    filtered_descriptions_tfidf = tfidf_description.transform(filtered_df['cleaned_description'])
    cosine_sim_description = cosine_similarity(input_description_tfidf, filtered_descriptions_tfidf)
    
    # Get top recommended tourist places based on description similarity
    top_indices_description = cosine_sim_description[0].argsort()[-5:][::-1]
    recommended_places = filtered_df.iloc[top_indices_description]

    # Prepare recommendations
    recommendations = recommended_places[['Place_Name', 'Description', 'Rating', 'City']].to_dict(orient='records')

    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)
