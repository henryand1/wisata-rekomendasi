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
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    text = text.strip()
    return text

# Preprocess descriptions
df['cleaned_description'] = df['Description'].apply(preprocess)

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

    # Transform user input for descriptions
    input_description_tfidf = tfidf_description.transform([preprocessed_input])

    # Calculate cosine similarity on descriptions
    cosine_sim_description = cosine_similarity(input_description_tfidf, tfidf_description_matrix)
    
    # Get top recommended tourist places based on description similarity
    top_indices_description = cosine_sim_description[0].argsort()[-5:][::-1]
    top_similarities = cosine_sim_description[0][top_indices_description]
    recommended_places = df.iloc[top_indices_description]

    # Prepare recommendations with similarity scores
    recommendations = []
    for idx, place in recommended_places.iterrows():
        recommendations.append({
            'Place_Name': place['Place_Name'],
            'Description': place['Description'],
            'Rating': place['Rating'],
            'City': place['City'],
            'Similarity_Score': top_similarities[len(recommendations)]
        })

    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)
