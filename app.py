from flask import Flask, request, render_template
from sentence_transformers import SentenceTransformer, util
import pickle

app = Flask(__name__)

# Load pre-trained model and classifier
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
with open('plagiarism_model.pkl', 'rb') as file:
    classifier = pickle.load(file)

# Function to compute similarity features
def get_similarity_feature(text1, text2):
    embedding1 = embedding_model.encode(text1, convert_to_tensor=True)
    embedding2 = embedding_model.encode(text2, convert_to_tensor=True)
    similarity = util.cos_sim(embedding1, embedding2).item()
    return [similarity]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/check', methods=['POST'])
def check_plagiarism():
    text1 = request.form['text1']
    text2 = request.form['text2']
    features = get_similarity_feature(text1, text2)
    prediction = classifier.predict([features])[0]
    result = "Plagiarized" if prediction == 1 else "Not Plagiarized"
    return render_template('result.html', text1=text1, text2=text2, result=result)

if __name__ == "__main__":
    app.run(debug=True)
