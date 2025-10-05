import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Title and author
st.title("🎬 IMDB Movie Review Sentiment Analysis")

# Sidebar
st.sidebar.title("About")
st.sidebar.markdown("This app predicts sentiment from movie reviews.")
st.sidebar.markdown("""
**Author:** [Kinson Vernet](https://kvernet.com)
""")


# Input
review = st.text_area("Enter your movie review here:", height=200)

if st.button("Predict sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        vector = vectorizer.transform([review])
        prediction = model.predict(vector)[0]
        st.success(f"🎯 Predicted sentiment: **{prediction.capitalize()}**")

# Footer
st.markdown(
    """
    <hr style="margin-top: 3em;">
    <div style="text-align: center;">
        <small>
            Made with ❤️ by <a href="https://kvernet.com" target="_blank">Kinson Vernet</a><br>
            🌐 <a href="https://kvernet.com" target="_blank">Website</a> |
            💻 <a href="https://github.com/kvernet" target="_blank">GitHub</a> |
            🔗 <a href="https://linkedin.com/in/kvernet" target="_blank">LinkedIn</a> |
            📧 <a href="mailto:kinson.vernet@gmail.com">Email</a>
        </small>
    </div>
    """,
    unsafe_allow_html=True
)
