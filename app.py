import streamlit as st
from predict import predict_ticker

st.set_page_config(page_title="Stock Predictor", layout="centered")

st.title("Stock Price Movement Predictor")
st.write("Enter any stock ticker")

ticker = st.text_input("Ticker").upper().strip()

if st.button("Predict"):
    if not ticker:
        st.error("Please enter a ticker")
    else:
        with st.spinner("Running model..."):
            try:
                result = predict_ticker(ticker)

                if result["prediction"] == 1:
                    st.success(f"{ticker} predicted UP 📈")
                else:
                    st.error(f"{ticker} predicted DOWN 📉")

                st.metric("Up Probability", f"{result['proba_up']*100:.2f}%")
                st.metric("Down Probability", f"{result['proba_down']*100:.2f}%")
                st.metric("Sentiment Score", f"{result['sentiment_score']:.3f}")
                st.metric("Combined Score", f"{result['combined_score']*100:.2f}%")

                st.subheader("News")

                if result["news_items"]:
                    for n in result["news_items"]:
                        title = n.get("title", "No title")
                        url = n.get("url", "")
                        source = n.get("source", "")

                        if url:
                            st.markdown(f"- [{title}]({url}) ({source})")
                        else:
                            st.write(f"- {title}")
                else:
                    st.write("No news found.")

            except Exception as e:
                st.error(f"Prediction failed: {e}")