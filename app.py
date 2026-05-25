import streamlit as st
from predict import predict_ticker

st.set_page_config(page_title="Stock Direction Predictor", layout="centered")

st.title("Stock Price Movement Predictor")
st.write("Predict whether a stock will go UP or DOWN next day.")

ticker = st.text_input("Enter stock ticker", value="AAPL").strip().upper()

if st.button("Predict"):
    try:
        result = predict_ticker(ticker)

        if result["prediction"] == 1:
            st.success(f"{ticker} predicted to go UP")
        else:
            st.error(f"{ticker} predicted to go DOWN")

        st.metric("Up Probability", f"{result['proba_up'] * 100:.2f}%")
        st.metric("Down Probability", f"{result['proba_down'] * 100:.2f}%")
        st.metric("Sentiment Score", f"{result['sentiment_score']:.3f}")
        st.metric("Combined Score", f"{result['combined_score'] * 100:.2f}%")

        with st.expander("News Headlines"):
            for h in result["headlines"]:
                st.write("- " + h)

    except Exception as e:
        st.error(str(e))