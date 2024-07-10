from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///instance/rekomendasi_tempat_wisata.sqlite'
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# db = SQLAlchemy(app)

# Memuat dataset
df = pd.read_csv('data/tourism_with_id.csv')

# Model untuk menyimpan history jawaban pengguna
# class History(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     user_input = db.Column(db.String(500))

#     def __init__(self, user_input):
#         self.user_input = user_input

# Endpoint untuk halaman utama
@app.route('/')
def home():
    return render_template('home.html')

# Endpoint untuk rekomendasi
@app.route('/recommend', methods=['POST'])
def recommend():
    user_input = request.form['user_input']
    user_answers = [user_input]

    # Simpan jawaban pengguna ke database
    # new_entry = History(user_input)
    # db.session.add(new_entry)
    # db.session.commit()

    # Menggabungkan deskripsi tempat wisata dan jawaban user
    descriptions = df['Description'].tolist() + user_answers

    # Menggunakan TF-IDF untuk representasi teks
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(descriptions)

    # Menghitung kemiripan kosinus
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

    # Mengurutkan tempat wisata berdasarkan kemiripan
    sim_scores = list(enumerate(cosine_sim[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Mendapatkan rekomendasi tempat wisata teratas
    top_sim_scores = sim_scores[:5]
    top_indices = [i[0] for i in top_sim_scores]

    # Menampilkan rekomendasi
    recommended_places = df.iloc[top_indices]
    recommendations = recommended_places[['Place_Name', 'Description', 'Rating']].to_dict(orient='records')

    return render_template('recommend.html', recommendations=recommendations)

if __name__ == '__main__':
    # db.create_all()  # Membuat database dan tabel
    app.run(debug=True)
