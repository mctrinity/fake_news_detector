import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# Streamlit UI
st.title("📰 Fake News Detector")
st.subheader("Enter a news article or statement to check if it's real or fake")

# User Input
user_input = st.text_area("Paste the news article here...", height=200)

if st.button("Check News"):
    if user_input.strip():
        # Process input
        transformed_input = vectorizer.transform([user_input])
        prediction = model.predict(transformed_input)[0]
        
        # Display result
        if prediction == 1:
            st.error("⚠️ This news is likely **FAKE**!")
        else:
            st.success("✅ This news appears to be **REAL**.")
    else:
        st.warning("Please enter a news article to analyze.")

# Footer
st.markdown("🔍 Built with **Machine Learning & NLP** | 🚀 Powered by Streamlit")
