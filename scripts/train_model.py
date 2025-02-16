import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the cleaned dataset
df = pd.read_csv("data/news_dataset_cleaned.csv")

# Drop any remaining missing values
df.dropna(subset=["text"], inplace=True)

# Ensure stratified sampling to maintain class balance
stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in stratified_split.split(df["text"], df["label"]):
    train_data = df.iloc[train_index]
    test_data = df.iloc[test_index]

X_train = train_data["text"]
y_train = train_data["label"]
X_test = test_data["text"]
y_test = test_data["label"]

# Convert text to numerical features using TF-IDF with n-grams
vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words="english", max_df=0.9, min_df=5)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train model with regularization to prevent overfitting
model = PassiveAggressiveClassifier(C=0.8, max_iter=2000)
model.fit(X_train_tfidf, y_train)

# Evaluate model
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Training Complete! Accuracy: {accuracy:.2f}")

# Display classification report
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model and vectorizer
joblib.dump(model, "models/fake_news_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("✅ Model and vectorizer saved successfully!")
