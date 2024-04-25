import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump

# Load the dataset
emails = pd.read_csv('emails.csv')

# Preprocess the data
emails['text'] = emails['text'].str.lower()
emails['text'] = emails['text'].str.replace('[^\w\s]', '')

# Create the TF-IDF vectorizer
vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(emails['text'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, emails['sentiment'], test_size=0.2)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the trained model to a file
dump(model, 'sentiment_model.joblib')

# Evaluate the model on the test set
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("The accuracy of the model is:", accuracy)