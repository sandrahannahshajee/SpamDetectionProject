import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Load dataset
data = pd.read_csv("dataset.csv")

# Convert labels
data['label'] = data['label'].map({'spam':1, 'ham':0})

# Convert text into numbers
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['message'])

y = data['label']

# Train model
model = MultinomialNB()
model.fit(X, y)

# Save model
pickle.dump(model, open("spam_model.pkl","wb"))
pickle.dump(vectorizer, open("vectorizer.pkl","wb"))

print("Model trained successfully")