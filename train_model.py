import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sentence_transformers import SentenceTransformer, util
import pickle

# Load dataset
data = pd.read_csv('plagiarism_dataset.csv')

# Pre-trained model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Feature extraction
def get_features(text1, text2):
    embedding1 = model.encode(text1, convert_to_tensor=True)
    embedding2 = model.encode(text2, convert_to_tensor=True)
    similarity = util.cos_sim(embedding1, embedding2).item()
    return [similarity]

# Prepare features and labels

features = [get_features(row['Text 1'], row['Text 2']) for _, row in data.iterrows()]
labels = data['Label'].values  # Assuming 'Label' is the column for labels


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train the classifier
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

# Evaluate the model
y_pred = classifier.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save the trained model
with open('plagiarism_model.pkl', 'wb') as file:
    pickle.dump(classifier, file)
print("Model saved as 'plagiarism_model.pkl'")
