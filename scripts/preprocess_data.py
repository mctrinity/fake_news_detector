import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already present
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def clean_text(text):
    """Cleans text while preserving important words."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove punctuation and numbers
    return text  # Keeping stopwords for better context understanding

# Load datasets
fake_df = pd.read_csv("data/Fake.csv")
true_df = pd.read_csv("data/True.csv")

# Add labels
fake_df["label"] = 1  # Fake news
true_df["label"] = 0  # Real news

# Merge datasets
df = pd.concat([fake_df, true_df], axis=0)

# Shuffle dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Apply text cleaning
df["text"] = df["text"].apply(clean_text)

# Drop empty rows
df.dropna(subset=["text"], inplace=True)

# Save cleaned dataset
df.to_csv("data/news_dataset_cleaned.csv", index=False)

print("✅ Dataset successfully merged, cleaned, and saved as news_dataset_cleaned.csv")
