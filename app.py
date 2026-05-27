import streamlit as st
from predict import predict_ticker

st.set_page_config(page_title="Stock Direction Predictor", layout="centered")

st.title("Stock Price Movement Predictor")
st.write("Predict whether a stock will go up or down next day.")

ticker = st.text_input("Enter stock ticker", value="AAPL").strip().upper()

if st.button("Predict"):
    if not ticker:
        st.error("Enter a ticker first.")
    else:
        try:
            result = predict_ticker(ticker)

            if result["prediction"] == 1:
                st.success(f"{result['ticker']} is predicted to go UP tomorrow.")
            else:
                st.error(f"{result['ticker']} is predicted to go DOWN tomorrow.")

            st.metric("Up Probability", f"{result['proba_up'] * 100:.2f}%")
            st.metric("Down Probability", f"{result['proba_down'] * 100:.2f}%")
            st.metric("Sentiment Score", f"{result['sentiment_score']:.3f}")
            st.metric("Combined Confidence", f"{result['combined_score'] * 100:.2f}%")

            with st.expander("News sources"):
                if result["news_items"]:
                    for item in result["news_items"]:
                        title = item.get("title", "Untitled")
                        source = item.get("source", "Unknown source")
                        url = item.get("url", "")

                        if url:
                            st.markdown(f"- [{title}]({url}) — {source}")
                        else:
                            st.write(f"- {title} — {source}")
                else:
                    st.write("No recent news found.")

        except Exception as e:
            st.error(f"Prediction failed: {e}")