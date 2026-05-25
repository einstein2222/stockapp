import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def get_sentiment(ticker):
    try:
        news = yf.Ticker(ticker).news or []
    except:
        return 0.0, []

    scores = []
    headlines = []

    for item in news[:10]:
        title = item.get("title", "")
        if title:
            headlines.append(title)
            scores.append(analyzer.polarity_scores(title)["compound"])

    if not scores:
        return 0.0, headlines

    return sum(scores) / len(scores), headlines