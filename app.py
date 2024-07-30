from flask import Flask, request, jsonify
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import re
import string
from nltk.corpus import stopwords
import nltk

app = Flask(__name__)

# Download stopwords
nltk.download('stopwords')

# Load preprocessed data and models
description_embeddings = torch.load('data/description_embeddings.pt')
cosine_sim_matrix = torch.load('data/collaboration_matrix.pt')

# Load IndoBERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
model = AutoModel.from_pretrained("indobenchmark/indobert-base-p1")

# Load the dataset (this assumes the dataset is in the same directory and format)
df = pd.read_csv('data/dataset-wisata-new.csv')

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    stop_words = set(stopwords.words('indonesian'))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    text = text.strip()
    return text

# Generate BERT embeddings
def generate_ner_embeddings(texts):
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

@app.route('/recommend', methods=['POST'])
def recommend():
    user_input = request.json.get('user_input')
    if not user_input:
        return jsonify({"error": "user_input is required"}), 400

    # Preprocess and split user input into multiple keywords/entities
    user_inputs = [preprocess_text(ui) for ui in user_input.split(',')]
    
    # Check preprocessed user inputs
    print("Preprocessed user inputs:", user_inputs)
    
    # Generate embeddings for each user input entity
    user_embeddings = [generate_ner_embeddings([ui])[0] for ui in user_inputs]
    
    # Ensure user_embeddings are correctly generated
    print("User embeddings:", user_embeddings)

    if len(user_embeddings) == 0:
        return jsonify({"error": "No embeddings generated for user input"}), 500

    # Initialize similarity scores
    total_similarities = torch.zeros(len(df))

    # Compute similarities for each user input entity
    for user_embedding in user_embeddings:
        similarities = cosine_similarity(user_embedding.reshape(1, -1), description_embeddings).flatten()
        total_similarities += torch.tensor(similarities)

    # Ensure there are no NaNs or Infs
    total_similarities = torch.where(torch.isfinite(total_similarities), total_similarities, torch.tensor(0.0))

    # # Check the contents of total_similarities
    # print("Total similarities:", total_similarities)

    if len(total_similarities) == 0:
        return jsonify({"error": "No similarity scores computed"}), 500

    # Get top 5 results
    try:
        # Ensure `total_similarities` is a 1D tensor
        if total_similarities.dim() != 1:
            total_similarities = total_similarities.flatten()
        
        # Get the indices of the top 5 similarities
        top_indices = total_similarities.argsort(descending=True)[:5]

        # Ensure top_indices is a list of integers
        if isinstance(top_indices, torch.Tensor):
            top_indices = top_indices.tolist()
    except ValueError as e:
        return jsonify({"error": f"Error in sorting similarities: {str(e)}"}), 500

    recommendations = []
    for idx in top_indices:
        try:
            recommendations.append({
                'Place_Name': df.iloc[idx]['Place_Name'],
                'Description': df.iloc[idx]['Description'],
                'Rating': df.iloc[idx]['Rating'],
                'City': df.iloc[idx]['City'],
                'Similarity_Score': float(total_similarities[idx])
            })
        except KeyError as e:
            return jsonify({"error": f"Key error: {str(e)}"}), 500
        except IndexError as e:
            return jsonify({"error": f"Index error: {str(e)}"}), 500

    return jsonify(recommendations)


if __name__ == '__main__':
    app.run(debug=True)
