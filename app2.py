import streamlit as st
import pickle

# Load model & vectorizer
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

st.set_page_config(page_title="Sentiment Analysis", page_icon="📝")

st.title("📝 Sentiment Analysis App")
st.markdown("### 🔍 Analyze Customer Sentiment")

user_input = st.text_area("Enter your review here:")

if st.button("Predict Sentiment"):

    if user_input.strip() != "":
        
        vector = tfidf.transform([user_input])

        prediction = model.predict(vector)[0]
        probability = model.predict_proba(vector).max()

        st.subheader("Result:")

        if prediction == "Positive":
            st.success("😊 Positive Review")
        elif prediction == "Negative":
            st.error("😡 Negative Review")
        else:
            st.info("😐 Neutral Review")

        st.write(f"Confidence Score: {round(probability * 100, 2)}%")

    else:
        st.warning("⚠️ Please enter a review!")

st.markdown("---")
st.caption("Built using Machine Learning & Streamlit 🚀")